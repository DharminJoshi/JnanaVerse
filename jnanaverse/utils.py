"""
JnanaVerse – utils.py
Shared utilities: checkpoint save/load, perplexity, seed setting, logging.
Developer(s): Dharmin Joshi / DevKay
"""

import os
import math
import random
import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
def get_logger(name: str = "JnanaVerse", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ──────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Union[str, Path],
    extra: dict | None = None,
):
    """Save model + optimizer state to a .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch":           epoch,
        "loss":            loss,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, str(path))
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: Union[str, Path],
    device: str = "cpu",
) -> dict:
    """Load model (and optionally optimizer) from a checkpoint file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"[Checkpoint] Loaded ← {path}  (epoch={ckpt.get('epoch')}, loss={ckpt.get('loss', 'N/A'):.4f})")
    return ckpt


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────
def perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity."""
    return math.exp(min(loss, 20))     # clamp to avoid overflow


def count_parameters(model: nn.Module, human_readable: bool = True) -> str | int:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not human_readable:
        return n
    for unit, label in [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")]:
        if n >= unit:
            return f"{n / unit:.2f}{label}"
    return str(n)


# ──────────────────────────────────────────────────────────────
# Device helper
# ──────────────────────────────────────────────────────────────
def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        dev = torch.device("cuda")
    elif prefer_gpu and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"[Device] Using: {dev}")
    return dev
