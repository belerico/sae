import os

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from sae import SaeConfig, SaeTrainer, TrainConfig
from sae.data import chunk_and_tokenize_streaming

if __name__ == "__main__":
    # Init DDP
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0
    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")
        print(f"Rank {rank}, Using DDP across {dist.get_world_size()} GPUs.")
        print(f"Rank {rank}, Local device: {torch.cuda.current_device()}")

    model_name = "EleutherAI/pythia-70m-deduped"
    l1_coefficient = 0.5
    max_seq_len = 512
    target_l0 = 64
    batch_size = 16
    lr = 7e-4

    dataset = load_dataset(
        "togethercomputer/RedPajama-Data-1T-Sample",
        split="train",
        trust_remote_code=True,
        streaming=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = chunk_and_tokenize_streaming(dataset, tokenizer, max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = AutoModel.from_pretrained(
        model_name,
        device_map={"": "cuda"},
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    sae_cfg = SaeConfig(
        expansion_factor=16,
        k=16,
        jumprelu=False,
        jumprelu_target_l0=target_l0,
        init_enc_as_dec_transpose=True,
        multi_topk=True,
    )
    cfg = TrainConfig(
        sae=sae_cfg,
        batch_size=batch_size,
        save_every=10000,
        lr=lr,
        lr_scheduler_name="cosine",
        lr_warmup_steps=0.01,
        l1_coefficient=l1_coefficient,
        l1_warmup_steps=0.1,
        max_seq_len=max_seq_len,
        use_l2_loss=True,
        num_training_tokens=1_000_000_000,
        normalize_activations=1,
        num_norm_estimation_tokens=1_000_000,
        run_name="checkpoints/{}-1024-lambda-{}-target-L0-{}-lr-{}".format(
            model_name,
            l1_coefficient,
            target_l0,
            lr,
        ),
        adam_betas=(0.0, 0.999),
        adam_epsilon=1e-8,
        distribute_modules=True,
        keep_last_n_checkpoints=2,
        auxk_alpha=1 / 32,
    )
    trainer = SaeTrainer(cfg, dataloader, model)
    trainer.fit()
