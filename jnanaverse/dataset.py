"""
JnanaVerse – dataset.py
PyTorch Dataset classes for language modelling and classification.
Developer(s): Dharmin Joshi / DevKay
"""

import torch
from torch.utils.data import Dataset
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Language-Modelling Dataset  (next-token prediction)
# ──────────────────────────────────────────────────────────────
class LMDataset(Dataset):
    """
    Concatenates all tokenised texts into one token stream, then
    yields (input, target) pairs of length `seq_len` with a 1-token shift.

    Args:
        texts:     List of raw strings.
        tokenizer: A `tokenizers.Tokenizer` instance.
        seq_len:   Context window length.
    """

    def __init__(self, texts: list[str], tokenizer, seq_len: int):
        self.seq_len = seq_len
        self.tokens: list[int] = []

        print("[LMDataset] Tokenising corpus …")
        for text in texts:
            enc = tokenizer.encode(text)
            self.tokens.extend(enc.ids)
        print(f"[LMDataset] Total tokens: {len(self.tokens):,}")

    def __len__(self) -> int:
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y


# ──────────────────────────────────────────────────────────────
# Seq2Seq Dataset  (for T5-style training)
# ──────────────────────────────────────────────────────────────
class Seq2SeqDataset(Dataset):
    """
    Dataset for seq2seq tasks (summarisation, translation, QA).

    Each sample is a (source_text, target_text) pair.
    Tokenisation is handled by a HuggingFace tokenizer (e.g. T5TokenizerFast).

    Args:
        pairs:         List of (source, target) string tuples.
        tokenizer:     HuggingFace tokenizer.
        max_src_len:   Maximum source token length.
        max_tgt_len:   Maximum target token length.
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer,
        max_src_len: int = 512,
        max_tgt_len: int = 128,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        src, tgt = self.pairs[idx]

        src_enc = self.tokenizer(
            src,
            max_length=self.max_src_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt_enc = self.tokenizer(
            tgt,
            max_length=self.max_tgt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = tgt_enc["input_ids"].squeeze()
        # Replace padding token id with -100 so loss ignores it
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      src_enc["input_ids"].squeeze(),
            "attention_mask": src_enc["attention_mask"].squeeze(),
            "labels":         labels,
        }


# ──────────────────────────────────────────────────────────────
# Classification Dataset
# ──────────────────────────────────────────────────────────────
class ClassificationDataset(Dataset):
    """
    Dataset for text classification.

    Args:
        texts:       List of raw strings.
        labels:      List of integer class labels.
        tokenizer:   HuggingFace tokenizer.
        max_length:  Maximum token length.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = 256,
    ):
        assert len(texts) == len(labels), "texts and labels must have the same length."
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }
