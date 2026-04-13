"""
JnanaVerse – tokenizer.py
BPE tokeniser utilities (build, load, encode, decode).
Developer(s): Dharmin Joshi / DevKay
"""

import os
from pathlib import Path
from typing import Union


def build_or_load_tokenizer(
    tokenizer_path: Union[str, Path] = "tokenizer.json",
    vocab_size: int = 10_000,
    training_files: list[str] | None = None,
    min_frequency: int = 2,
):
    """
    Load an existing HuggingFace-compatible BPE tokeniser from disk,
    or train a new one on `training_files` and save it.

    Args:
        tokenizer_path: Where to save / load the tokeniser JSON.
        vocab_size:     Target vocabulary size.
        training_files: List of plain-text files to train on (only needed
                        when the tokeniser doesn't exist yet).
        min_frequency:  Minimum token frequency to be included in vocab.

    Returns:
        A `tokenizers.Tokenizer` instance.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    tokenizer_path = Path(tokenizer_path)

    if tokenizer_path.exists():
        print(f"[Tokenizer] Loading from '{tokenizer_path}' …")
        return Tokenizer.from_file(str(tokenizer_path))

    if not training_files:
        raise ValueError(
            "Tokeniser not found and no `training_files` provided for training."
        )

    print(f"[Tokenizer] Training new BPE tokeniser (vocab_size={vocab_size}) …")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True,
    )
    tokenizer.train(training_files, trainer)

    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    print(f"[Tokenizer] Saved to '{tokenizer_path}' (vocab={tokenizer.get_vocab_size()}).")
    return tokenizer


def write_temp_corpus(texts: list[str], path: str = "temp_corpus.txt") -> str:
    """Write a list of strings to a temporary flat-text file for tokeniser training."""
    with open(path, "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line.strip() + "\n")
    return path
