import os
from contextlib import nullcontext, redirect_stdout
from typing import cast

import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from safetensors.torch import load_model
from simple_parsing import parse
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel

from .config import RunConfig
from .data import MemmapDataset, chunk_and_tokenize, chunk_and_tokenize_streaming
from .trainer import SaeTrainer


def load_artifacts(
    args: RunConfig, rank: int
) -> tuple[PreTrainedModel, Dataset | IterableDataset | MemmapDataset]:
    ddp = os.environ.get("LOCAL_RANK") is not None

    if args.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = AutoModel.from_pretrained(
        args.model,
        device_map={"": f"cuda:{rank}"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=args.load_in_8bit) if args.load_in_8bit else None
        ),
        revision=args.revision,
        torch_dtype=dtype,
        token=args.hf_token,
    )

    # For memmap-style datasets
    if args.dataset.endswith(".bin"):
        dataset = MemmapDataset(args.dataset, args.max_seq_len, args.max_examples)
    else:
        # For Huggingface datasets
        try:
            dataset = load_dataset(
                args.dataset,
                name=args.dataset_name,
                split=args.split,
                # TODO: Maybe set this to False by default? But RPJ requires it.
                trust_remote_code=True,
                streaming=args.streaming,
            )
        except ValueError as e:
            # Automatically use load_from_disk if appropriate
            if "load_from_disk" in str(e):
                dataset = Dataset.load_from_disk(args.dataset, keep_in_memory=False)
            else:
                raise e
        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            raise ValueError("DatasetDict and IterableDatasetDict datasets are supported for now.")

        # Shard the dataset across all ranks
        if ddp and args.streaming:
            dataset = dataset.shard(dist.get_world_size(), rank)

        # assert isinstance(dataset, Dataset)
        column_names = dataset.column_names
        if column_names is None:
            raise ValueError("Dataset does not have column names.")
        if "input_ids" not in column_names:
            tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
            if args.streaming:
                dataset = cast(IterableDataset, dataset)
                dataset = chunk_and_tokenize_streaming(
                    dataset,
                    tokenizer,
                    max_seq_len=args.max_seq_len,
                    return_final_batch=False,
                )
            else:
                dataset = cast(Dataset, dataset)
                dataset = chunk_and_tokenize(
                    dataset,
                    tokenizer,
                    max_seq_len=args.max_seq_len,
                    num_proc=args.data_preprocessing_num_proc,
                    return_final_batch=False,
                )
        else:
            print("Dataset already tokenized; skipping tokenization.")

        print(f"Shuffling dataset with seed {args.seed}")
        dataset = dataset.shuffle(args.seed)
    return model, dataset


def run():
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    args = parse(RunConfig)

    # Awkward hack to prevent other ranks from duplicating data preprocessing
    if not ddp or rank == 0:
        model, dataset = load_artifacts(args, rank)
    if ddp:
        dist.barrier()
        if rank != 0:
            model, dataset = load_artifacts(args, rank)
        if not args.streaming or args.dataset.endswith(".bin"):
            dataset = dataset.shard(dist.get_world_size(), rank)

    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        print(f"Training on '{args.dataset}' (split '{args.split}')")
        print(f"Storing model weights in {model.dtype}")

        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        trainer = SaeTrainer(args, dataloader, model)
        if args.resume:
            trainer.load_state(args.run_name or "sae-ckpts")
        elif args.finetune:
            for name, sae in trainer.saes.items():
                load_model(
                    sae, f"{args.finetune}/{name}/sae.safetensors", device=str(model.device)
                )

        trainer.fit()


if __name__ == "__main__":
    run()
