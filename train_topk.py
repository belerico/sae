import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel

from sae import SaeConfig, SaeTrainer, TrainConfig

if __name__ == "__main__":
    model_name = "EleutherAI/pythia-160m-deduped"
    l1_coefficient = 0.0
    max_seq_len = 1024
    target_l0 = None
    batch_size = 4
    lr = 12e-4
    k = 128

    # Streaming dataset example
    # dataset = load_dataset(
    #     "allenai/c4",
    #     "en",
    #     split="train",
    #     trust_remote_code=True,
    #     streaming=True,
    # )
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
            "input_ids": torch.stack(list(torch.tensor(example["tokens"]) for example in x), dim=0)
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
            k=k,
            jumprelu=False,
            multi_topk=False,
            init_b_dec_as_zeros=False,
            init_enc_as_dec_transpose=True,
        ),
        batch_size=batch_size,
        save_every=100_000,
        layers=list(range(12)),
        lr=lr,
        lr_scheduler_name="constant",
        lr_warmup_steps=0.0,
        l1_coefficient=l1_coefficient,
        l1_warmup_steps=0.0,
        max_seq_len=max_seq_len,
        use_l2_loss=True,
        cycle_iterator=True,
        num_training_tokens=1_000_000_000,
        normalize_activations=1.0,
        num_norm_estimation_tokens=2_000_000,
        run_name="checkpoints/{}-1024-topk-{}-lambda-{}-target-L0-{}-lr-{}".format(
            model_name, k, l1_coefficient, target_l0, lr
        ),
        adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8,
        auxk_alpha=1 / 32,
    )
    trainer = SaeTrainer(cfg, data_loader, model)
    trainer.fit()
