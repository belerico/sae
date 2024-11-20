import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

from sae import SaeConfig, SaeTrainer, TrainConfig
from sae.data import chunk_and_tokenize, chunk_and_tokenize_streaming

if __name__ == "__main__":
    model_name = "EleutherAI/pythia-70m-deduped"
    max_seq_len = 1024
    batch_size = 4

    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        trust_remote_code=True,
        streaming=True,
    )
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # dataset = chunk_and_tokenize_streaming(dataset, tokenizer, max_seq_len=max_seq_len)
    dataset = load_dataset(
        "NeelNanda/pile-small-tokenized-2b",
        streaming=True,
        split="train",
        trust_remote_code=True,
    )

    def from_tokens(x):
        return {
            "input_ids": torch.stack(
                list(torch.tensor(example["tokens"]) for example in x), dim=0
            )
        }

    data_loader = DataLoader(
        dataset,
        collate_fn=from_tokens,
        batch_size=batch_size,
    )
    model = AutoModel.from_pretrained(
        model_name,
        device_map={"": "cuda"},
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    cfg = TrainConfig(
        SaeConfig(
            expansion_factor=16,
            k=-1,
            jumprelu=True,
            init_enc_as_dec_transpose=True,
        ),
        batch_size=batch_size,
        save_every=25_000,
        layers=[3],
        lr=7e-5,
        lr_init=7e-6,
        lr_end=7e-6,
        lr_scheduler_name="constant",
        lr_warmup_steps=0.05,
        lr_decay_steps=0.95,
        l1_coefficient=1e-3,
        l1_warmup_steps=0.5,
        max_seq_len=max_seq_len,
        use_l2_loss=True,
        cycle_iterator=True,
        num_training_tokens=1_000_000_000,
        normalize_activations=1,
        num_norm_estimation_tokens=2_000_000,
        run_name="pythia-70m-deduped-1024-lambda-0.5-lr-7e-5",
        adam_betas=(0.0, 0.999),
        adam_epsilon=1e-8,
    )
    trainer = SaeTrainer(cfg, data_loader, model)
    trainer.fit()
