import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

from sae import SaeConfig, SaeTrainer, TrainConfig
from sae.data import chunk_and_tokenize, chunk_and_tokenize_streaming

if __name__ == "__main__":
    model_name = "EleutherAI/pythia-70m-deduped"
    total_tokens = 1_000_000
    max_seq_len = 1024
    batch_size = 2

    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        trust_remote_code=True,
        streaming=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = chunk_and_tokenize_streaming(dataset, tokenizer, max_seq_len=max_seq_len)
    data_loader = DataLoader(
        dataset,
        collate_fn=DataCollatorWithPadding(tokenizer),
        batch_size=batch_size,
    )
    model = AutoModel.from_pretrained(
        model_name,
        device_map={"": "mps"},
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    cfg = TrainConfig(
        SaeConfig(
            expansion_factor=16,
            k=-1,
            jumprelu=True,
            init_enc_as_dec_transpose=True,
            equiangular_init=True,
        ),
        batch_size=batch_size,
        save_every=25_000,
        layers=[3, 4],
        lr=1e-3,
        lr_scheduler_name="constant",
        lr_warmup_steps=0.0005,
        lr_decay_steps=0.0,
        l1_coefficient=1.0,
        l1_warmup_steps=0.005,
        max_seq_len=max_seq_len,
        use_l2_loss=True,
        cycle_iterator=True,
        num_training_tokens=10_000_000,
        normalize_activations="expected_norm_1",
        log_to_wandb=True
    )
    trainer = SaeTrainer(cfg, data_loader, model)
    trainer.fit()
