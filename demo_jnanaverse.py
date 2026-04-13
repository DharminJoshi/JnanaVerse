"""
JnanaVerse – demo_jnanaverse.py
Demonstrates the full JnanaVerse model:
  1. Text generation   (T5 seq2seq)
  2. Classification    (encoder + MLP head)
  3. Sentence similarity (cosine)
  4. Adapter integration (optional, requires `adapters` package)

Run:
    python demo_jnanaverse.py
Developer(s): Dharmin Joshi / DevKay
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jnanaverse import (
    JnanaVerse,
    Seq2SeqDataset,
    ClassificationDataset,
    train_seq2seq,
    train_classification,
    build_optimizer_scheduler,
    set_seed,
    get_device,
    count_parameters,
)


def demo_generation(model: JnanaVerse, device: str):
    print("\n" + "=" * 60)
    print("  TASK 1 – Text Generation")
    print("=" * 60)
    prompts = [
        "translate English to Sanskrit: The universe is vast and beautiful.",
        "summarize: JnanaVerse is an open-source NLP framework that supports text generation, classification, and similarity.",
        "question: What is transformers used for? context: Transformers are used for sequence modelling in NLP.",
    ]
    for p in prompts:
        out = model.generate_text(p, max_new_tokens=64, device=device)
        print(f"  Prompt : {p[:80]}…")
        print(f"  Output : {out}\n")


def demo_classification(model: JnanaVerse, device: str):
    print("\n" + "=" * 60)
    print("  TASK 2 – Text Classification  (untrained head – random)")
    print("=" * 60)
    texts = [
        "I love this product, it is amazing!",
        "This is terrible, very disappointed.",
        "It's okay, nothing special.",
    ]
    labels = [0, 1, 2]   # dummy classes: positive, negative, neutral

    dataset    = ClassificationDataset(texts, labels, model.tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=2)

    # One quick forward pass (not trained – just demonstrating the API)
    model.eval()
    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask, task="classification")
        preds = logits.argmax(dim=-1).tolist()
        print(f"  Logits : {logits.cpu().round(decimals=2).tolist()}")
        print(f"  Preds  : {preds}\n")
        break


def demo_similarity(model: JnanaVerse, device: str):
    print("\n" + "=" * 60)
    print("  TASK 3 – Sentence Similarity")
    print("=" * 60)
    pairs = [
        ("I love machine learning.", "Deep learning is my passion."),
        ("The cat sat on the mat.",  "The sun is shining brightly."),
    ]
    model.eval()
    for s1, s2 in pairs:
        enc1 = model.encode_text(s1, device=device)
        enc2 = model.encode_text(s2, device=device)
        with torch.no_grad():
            score = model(
                input_ids    =(enc1["input_ids"],      enc2["input_ids"]),
                attention_mask=(enc1["attention_mask"], enc2["attention_mask"]),
                task="similarity",
            )
        print(f"  '{s1}'")
        print(f"  '{s2}'")
        print(f"  Cosine similarity: {score.item():.4f}\n")


def demo_adapter(model: JnanaVerse):
    print("\n" + "=" * 60)
    print("  TASK 4 – Adapter Integration (optional)")
    print("=" * 60)
    try:
        model.add_adapter("summarisation", reduction_factor=16)
        model.enable_adapter("summarisation")
        print("  Adapter added and activated successfully.")
        model.disable_adapters()
    except ImportError as e:
        print(f"  Skipped (adapters package not installed): {e}")


def main():
    set_seed(42)
    device = get_device()

    print("\n╔══════════════════════════════════════════════╗")
    print("║          JnanaVerse v2.0 – Demo              ║")
    print("║  Developer(s): Dharmin Joshi / DevKay        ║")
    print("╚══════════════════════════════════════════════╝")

    model = JnanaVerse(model_name="t5-small", num_classes=3).to(device)
    print(f"[Main] Total trainable parameters: {count_parameters(model)}")

    demo_generation(model, device)
    demo_similarity(model, device)
    demo_classification(model, device)
    demo_adapter(model)

    print("\n✓ Demo complete.")


if __name__ == "__main__":
    main()
