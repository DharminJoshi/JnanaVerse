# JnanaVerse v2.0

**Open-Source Multi-Task NLP Framework**  
*Developer(s): Dharmin Joshi / DevKay*

---

## Overview

JnanaVerse is a fully open-source, modular NLP framework that integrates:

- **T5-based multi-task model** (HuggingFace) for generation, classification, and similarity
- **Custom Transformer LM** trainable from scratch with sinusoidal positional encoding
- **BPE Tokenizer** (trainable on custom corpora via `tokenizers`)
- **Adapter fine-tuning** via the `adapters` package (Pfeiffer architecture)
- **Flexible training utilities** with AMP, gradient clipping, cosine LR scheduling

No paid APIs or commercial services are required.

---

## Project Structure

```
jnanaverse/
├── jnanaverse/
│   ├── __init__.py       # Public API exports
│   ├── model.py          # JnanaVerse (T5) + CustomTransformerLM
│   ├── tokenizer.py      # BPE tokenizer build/load
│   ├── dataset.py        # LMDataset, Seq2SeqDataset, ClassificationDataset
│   ├── trainer.py        # Training loops + optimizer/scheduler factory
│   └── utils.py          # Checkpointing, seeding, metrics, device helpers
├── train_custom_lm.py    # Train CustomTransformerLM from scratch
├── demo_jnanaverse.py    # Full demo: generation, similarity, classification, adapters
├── tokenizer.json        # Saved BPE tokenizer (auto-generated on first run)
├── requirements.txt
└── setup.py
```

---

## Installation

```bash
pip install -r requirements.txt

# Optional adapter support
pip install adapters
```

---

## Quick Start

### 1. CLI Chatbot (interactive terminal)
```bash
python chatbot.py
# Options:
python chatbot.py --model t5-base --device cuda
```
Commands inside the chat: `/generate`, `/classify`, `/similarity`, `/help`, `/exit`

### 2. Web Chatbot (browser UI)
```bash
pip install flask
python chatbot_web.py
# Opens http://localhost:5000 automatically
# Options:
python chatbot_web.py --model t5-base --port 8080
```

### 3. Train the custom LM from scratch
```bash
python train_custom_lm.py
```

### 4. Run the full JnanaVerse demo
```bash
python demo_jnanaverse.py
```

### 3. Use as a library

```python
from jnanaverse import JnanaVerse, set_seed, get_device

set_seed(42)
device = get_device()

model = JnanaVerse(model_name="t5-small", num_classes=5).to(device)

# Text generation
text = model.generate_text(
    "translate English to Sanskrit: Hello, world!",
    device=device,
)
print(text)

# Sentence similarity
enc1 = model.encode_text("I love NLP.", device=device)
enc2 = model.encode_text("Natural language processing is great.", device=device)
score = model(
    input_ids=(enc1["input_ids"], enc2["input_ids"]),
    attention_mask=(enc1["attention_mask"], enc2["attention_mask"]),
    task="similarity",
)
print(f"Similarity: {score.item():.4f}")

# Adapter fine-tuning (requires `pip install adapters`)
model.add_adapter("my_task", reduction_factor=16)
model.enable_adapter("my_task")
```

---

## Key Features

| Feature | Details |
|---|---|
| **Multi-task** | Generation, classification, sentence similarity |
| **Adapters** | Pfeiffer-style via `adapters` package |
| **Custom LM** | Train decoder-only Transformer from scratch |
| **Tokenizer** | BPE, trainable on any plain-text corpus |
| **Training** | AMP, gradient clipping, warmup + cosine LR |
| **Checkpointing** | Save/load model + optimizer state |
| **Weight tying** | Embedding ↔ LM head (reduces params) |
| **Pre-LayerNorm** | Stabilises deep network training |

---

## Supported Datasets (open)

- Wikipedia dumps
- Common Crawl (C4)
- Project Gutenberg
- OpenWebText
- Any custom plain-text corpus

---

## 📄 License

MIT [LICENSE](LICENSE) — use it, modify it, ship it.

---
