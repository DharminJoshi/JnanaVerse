"""
JnanaVerse – trainer.py
Flexible training loop supporting LM pre-training, seq2seq fine-tuning,
and classification fine-tuning. Includes gradient clipping, LR scheduling,
and optional mixed-precision (AMP) training.
Developer(s): Dharmin Joshi / DevKay
"""

import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast


# ──────────────────────────────────────────────────────────────
# LM / Custom Transformer Trainer
# ──────────────────────────────────────────────────────────────
def train_lm(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
    grad_clip: float = 1.0,
    use_amp: bool = False,
    epoch: int = 1,
) -> float:
    """One full training epoch for a causal language model."""
    model.train()
    total_loss = 0.0
    scaler = GradScaler(enabled=use_amp)
    start = time.time()

    for step, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(inputs)                        # (B, T, V)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,                           # skip PAD
            )

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if step % 50 == 0:
            elapsed = time.time() - start
            print(
                f"  [Epoch {epoch}] step={step:>5} | loss={loss.item():.4f} | "
                f"elapsed={elapsed:.1f}s"
            )

    avg_loss = total_loss / max(len(dataloader), 1)
    print(f"  ✓ Epoch {epoch} avg loss: {avg_loss:.4f}")
    return avg_loss


# ──────────────────────────────────────────────────────────────
# Seq2Seq Trainer  (JnanaVerse / T5)
# ──────────────────────────────────────────────────────────────
def train_seq2seq(
    model,                     # JnanaVerse instance
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
    grad_clip: float = 1.0,
    use_amp: bool = False,
    epoch: int = 1,
) -> float:
    """One full training epoch for the JnanaVerse seq2seq (generation) task."""
    model.train()
    total_loss = 0.0
    scaler = GradScaler(enabled=use_amp)

    for step, batch in enumerate(dataloader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                task="generation",
            )
            loss = out.loss

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if step % 20 == 0:
            print(f"  [Epoch {epoch}] step={step:>5} | seq2seq loss={loss.item():.4f}")

    avg_loss = total_loss / max(len(dataloader), 1)
    print(f"  ✓ Epoch {epoch} seq2seq avg loss: {avg_loss:.4f}")
    return avg_loss


# ──────────────────────────────────────────────────────────────
# Classification Trainer  (JnanaVerse)
# ──────────────────────────────────────────────────────────────
def train_classification(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
    grad_clip: float = 1.0,
    use_amp: bool = False,
    epoch: int = 1,
) -> float:
    """One full training epoch for the classification head."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    scaler = GradScaler(enabled=use_amp)

    for step, batch in enumerate(dataloader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task="classification",
            )
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if step % 20 == 0:
            acc = correct / max(total, 1)
            print(
                f"  [Epoch {epoch}] step={step:>5} | cls loss={loss.item():.4f} | acc={acc:.3f}"
            )

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1)
    print(f"  ✓ Epoch {epoch} cls avg loss: {avg_loss:.4f}  accuracy: {accuracy:.3f}")
    return avg_loss


# ──────────────────────────────────────────────────────────────
# Convenience: build an AdamW + cosine scheduler pair
# ──────────────────────────────────────────────────────────────
def build_optimizer_scheduler(
    model: nn.Module,
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    total_steps: int = 1000,
    warmup_steps: int = 100,
) -> tuple:
    """
    Returns (optimizer, scheduler).
    Uses linear warmup then cosine decay.
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159265 * progress)).item())

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
