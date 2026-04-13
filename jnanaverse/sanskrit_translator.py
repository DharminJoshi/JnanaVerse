"""
JnanaVerse – sanskrit_translator.py
English → Sanskrit translation using Meta's NLLB-200 multilingual model.

Model: facebook/nllb-200-distilled-600M
  - Supports 200+ languages including Sanskrit (san_Deva – Devanagari script)
  - ~2.4 GB download on first use (cached locally by HuggingFace)
  - Lighter option: facebook/nllb-200-distilled-1.3B  (better quality, ~5 GB)

Usage:
    from jnanaverse.sanskrit_translator import SanskritTranslator
    translator = SanskritTranslator()
    print(translator.translate("The sky is vast and full of stars."))

Developer(s): Dharmin Joshi / DevKay
"""

import torch
from typing import Union


# NLLB language codes
_SRC_LANG = "eng_Latn"   # English  (Latin script)
_TGT_LANG = "san_Deva"   # Sanskrit (Devanagari script)

_DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
_QUALITY_MODEL = "facebook/nllb-200-distilled-1.3B"   # better but larger


class SanskritTranslator:
    """
    English → Sanskrit translator backed by Meta NLLB-200.

    Args:
        model_name: HuggingFace model ID. Defaults to the 600M distilled variant.
        device:     'cpu', 'cuda', 'mps', or 'auto' (picks best available).
        max_length: Maximum output token length.
        num_beams:  Beam width for beam search.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "auto",
        max_length: int = 256,
        num_beams: int = 5,
    ):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.max_length = max_length
        self.num_beams  = num_beams

        print(f"[SanskritTranslator] Loading '{model_name}' on {self.device} …")
        print("  (First run: ~2.4 GB download, cached after that)")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            src_lang=_SRC_LANG,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Resolve the Sanskrit target language token id
        self._tgt_lang_id = self.tokenizer.lang_code_to_id[_TGT_LANG]
        print(f"[SanskritTranslator] Ready. Sanskrit token id: {self._tgt_lang_id}")

    @torch.no_grad()
    def translate(self, text: str) -> str:
        """
        Translate a single English string to Sanskrit (Devanagari).

        Args:
            text: English input text.
        Returns:
            Sanskrit translation as a Unicode string.
        """
        text = text.strip()
        if not text:
            return ""

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            forced_bos_token_id=self._tgt_lang_id,
            max_length=self.max_length,
            num_beams=self.num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def translate_batch(self, texts: list[str], batch_size: int = 8) -> list[str]:
        """
        Translate a list of English strings to Sanskrit.

        Args:
            texts:      List of English strings.
            batch_size: Number of sentences per forward pass.
        Returns:
            List of Sanskrit translations (same order as input).
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = [t.strip() for t in texts[i : i + batch_size]]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            output_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=self._tgt_lang_id,
                max_length=self.max_length,
                num_beams=self.num_beams,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

            for ids in output_ids:
                results.append(self.tokenizer.decode(ids, skip_special_tokens=True))

        return results

    def __repr__(self) -> str:
        return (
            f"SanskritTranslator("
            f"model='{self.model.config._name_or_path}', "
            f"device={self.device}, "
            f"beams={self.num_beams})"
        )


# ──────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    translator = SanskritTranslator()

    test_sentences = [
        "The universe is vast and full of stars.",
        "Knowledge is the greatest wealth.",
        "Peace is the foundation of all happiness.",
        "I am an AI built with transformers.",
        "The river flows gently through the forest.",
    ]

    print("\n" + "═" * 60)
    print("  English → Sanskrit Translations")
    print("═" * 60)
    for eng in test_sentences:
        san = translator.translate(eng)
        print(f"\n  EN: {eng}")
        print(f"  SA: {san}")
    print()
