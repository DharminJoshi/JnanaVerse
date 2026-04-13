"""
JnanaVerse – train_custom_lm.py
Train the CustomTransformerLM from scratch on a small demo corpus,
then generate text using top-k / nucleus sampling.

Run:
    python train_custom_lm.py
Developer(s): Dharmin Joshi / DevKay
"""

import os
import torch
from torch.utils.data import DataLoader

from jnanaverse import (
    CustomTransformerLM,
    LMDataset,
    build_or_load_tokenizer,
    write_temp_corpus,
    train_lm,
    build_optimizer_scheduler,
    set_seed,
    get_device,
    save_checkpoint,
    perplexity,
    count_parameters,
)


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
CFG = dict(
    # Model
    vocab_size    = 10_000,
    d_model       = 256,
    n_layers      = 4,
    n_heads        = 8,
    seq_len        = 32,
    ff_dim         = 1024,
    dropout        = 0.1,
    # Training
    batch_size     = 64,
    epochs         = 5,
    lr             = 5e-4,
    weight_decay   = 0.01,
    grad_clip      = 1.0,
    use_amp        = False,   # Set True if CUDA is available
    # Paths
    tokenizer_path = "tokenizer.json",
    ckpt_path      = "checkpoints/custom_lm.pt",
    # Generation
    prompt         = "Hello",
    max_new_tokens = 40,
    temperature    = 0.85,
    top_k          = 40,
    top_p          = 0.9,
    seed           = 42,
)


def main():
    set_seed(CFG["seed"])
    device = get_device()

    # ── Demo corpus ──────────────────────────────────────────
    corpus = [
        "Hello, how are you?",
        "I am fine, thank you!",
        "What is your name?",
        "I am an AI chatbot built with transformers.",
        "Nice to meet you!",
        "How can I help you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is a fascinating field.",
        "Transformers have revolutionised machine learning.",
        "Open-source tools empower researchers worldwide.",
    ] * 500    # repeat for a reasonable corpus size

    # ── Tokeniser ────────────────────────────────────────────
    if not os.path.exists(CFG["tokenizer_path"]):
        tmp = write_temp_corpus(corpus, "temp_corpus.txt")
        tokenizer = build_or_load_tokenizer(
            tokenizer_path=CFG["tokenizer_path"],
            vocab_size=CFG["vocab_size"],
            training_files=[tmp],
        )
        os.remove(tmp)
    else:
        tokenizer = build_or_load_tokenizer(tokenizer_path=CFG["tokenizer_path"])

    actual_vocab = tokenizer.get_vocab_size()
    print(f"[Main] Vocabulary size: {actual_vocab:,}")

    # ── Dataset & DataLoader ─────────────────────────────────
    dataset    = LMDataset(corpus, tokenizer, seq_len=CFG["seq_len"])
    dataloader = DataLoader(
        dataset, batch_size=CFG["batch_size"], shuffle=True, num_workers=0
    )

    # ── Model ────────────────────────────────────────────────
    model = CustomTransformerLM(
        vocab_size=actual_vocab,
        d_model   =CFG["d_model"],
        n_layers  =CFG["n_layers"],
        n_heads   =CFG["n_heads"],
        seq_len   =CFG["seq_len"],
        ff_dim    =CFG["ff_dim"],
        dropout   =CFG["dropout"],
    ).to(device)

    print(f"[Main] Model parameters: {count_parameters(model)}")

    total_steps = CFG["epochs"] * len(dataloader)
    optimizer, scheduler = build_optimizer_scheduler(
        model,
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
        total_steps=total_steps,
        warmup_steps=max(1, total_steps // 10),
    )

    # ── Training loop ────────────────────────────────────────
    for epoch in range(1, CFG["epochs"] + 1):
        avg_loss = train_lm(
            model, dataloader, optimizer, device,
            scheduler=scheduler,
            grad_clip=CFG["grad_clip"],
            use_amp=CFG["use_amp"],
            epoch=epoch,
        )
        ppl = perplexity(avg_loss)
        print(f"  → Perplexity: {ppl:.2f}")

    save_checkpoint(model, optimizer, CFG["epochs"], avg_loss, CFG["ckpt_path"])

    # ── Text generation ──────────────────────────────────────
    print(f"\n[Main] Generating from prompt: '{CFG['prompt']}'")
    prompt_ids = torch.tensor(
        [tokenizer.encode(CFG["prompt"]).ids], dtype=torch.long, device=device
    )
    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=CFG["max_new_tokens"],
        temperature=CFG["temperature"],
        top_k=CFG["top_k"],
        top_p=CFG["top_p"],
    )
    generated = tokenizer.decode(output_ids[0].tolist())
    print("[Generated]:", generated)


if __name__ == "__main__":
    main()
