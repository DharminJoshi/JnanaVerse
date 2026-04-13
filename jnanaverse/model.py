"""
JnanaVerse – model.py
Multi-task NLP: generation, classification, similarity.
Developer(s): Dharmin Joshi / DevKay
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Positional Encoding  (sinusoidal, as in "Attention is All You Need")
# ──────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


# ──────────────────────────────────────────────────────────────
# Custom Autoregressive Transformer Language Model
# ──────────────────────────────────────────────────────────────
class CustomTransformerLM(nn.Module):
    """
    Decoder-only (causal) Transformer LM trainable from scratch.
    Features:
      - Pre-LayerNorm (stabilises training)
      - Weight tying between token embedding and output projection
      - top-k / top-p (nucleus) sampling via generate()
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        seq_len: int = 128,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # Weight tying

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T)  →  logits: (B, T, vocab_size)"""
        T = x.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()
        emb = self.pos_enc(self.token_emb(x) * math.sqrt(self.d_model))
        out = self.transformer(emb, mask=causal_mask, is_causal=True)
        return self.lm_head(self.ln_f(out))

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Auto-regressively sample tokens given a prompt."""
        self.eval()
        ctx = prompt_ids.clone()
        for _ in range(max_new_tokens):
            ctx_crop = ctx[:, -self.seq_len :]
            logits = self(ctx_crop)[:, -1, :] / max(temperature, 1e-8)

            # Top-k filter
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filter
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cumprobs - F.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            ctx = torch.cat([ctx, next_tok], dim=1)
        return ctx

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────
# JnanaVerse  – T5-based Multi-Task Model
# ──────────────────────────────────────────────────────────────
class JnanaVerse(nn.Module):
    """
    Modular NLP framework wrapping HuggingFace T5ForConditionalGeneration.

    Tasks:
      • 'generation'      – seq2seq text generation / summarisation / translation
      • 'classification'  – text classification via encoder + MLP head
      • 'similarity'      – cosine sentence similarity via encoder CLS vectors

    Optional lightweight adapter fine-tuning via the `adapters` package.
    """

    def __init__(self, model_name: str = "t5-base", num_classes: int = 10):
        super().__init__()
        from transformers import T5ForConditionalGeneration, T5TokenizerFast

        print(f"[JnanaVerse] Loading '{model_name}' from HuggingFace Hub …")
        self.base_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer  = T5TokenizerFast.from_pretrained(model_name)
        self.d_model    = self.base_model.config.d_model
        self._adapters: dict[str, bool] = {}

        # Classification head (mean-pooled encoder → MLP)
        self.cls_head = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

        # Similarity head
        self.sim_head = nn.CosineSimilarity(dim=1)

    # ── Forward ───────────────────────────────────────────────
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
        task: str = "generation",
    ):
        if task == "generation":
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True,
            )

        elif task == "classification":
            enc_out = self.base_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state                        # (B, T, D)

            # Mean pool over non-padding positions
            mask_exp = (
                attention_mask.unsqueeze(-1).float()
                if attention_mask is not None
                else torch.ones(*enc_out.shape[:2], 1, device=enc_out.device)
            )
            pooled = (enc_out * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
            return self.cls_head(pooled)               # (B, num_classes)

        elif task == "similarity":
            # input_ids / attention_mask are 2-tuples
            def _encode(ids, mask):
                return self.base_model.encoder(
                    input_ids=ids, attention_mask=mask, return_dict=True
                ).last_hidden_state[:, 0, :]           # CLS-style: first token

            v1 = _encode(input_ids[0], attention_mask[0] if attention_mask else None)
            v2 = _encode(input_ids[1], attention_mask[1] if attention_mask else None)
            return self.sim_head(v1, v2)               # (B,)

        else:
            raise ValueError(
                f"Unknown task '{task}'. Choose: 'generation', 'classification', 'similarity'."
            )

    # ── Text generation (beam search / greedy) ────────────────
    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        device: str = "cpu",
    ) -> str:
        self.eval()
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        out_ids = self.base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
        )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    # ── Adapter helpers ───────────────────────────────────────
    def add_adapter(self, task_name: str, reduction_factor: int = 16):
        try:
            from adapters import AdapterConfig
        except ImportError:
            raise ImportError("Run: pip install adapters")

        cfg = AdapterConfig.load("pfeiffer", reduction_factor=reduction_factor)
        self.base_model.add_adapter(task_name, config=cfg)
        self.base_model.train_adapter(task_name)
        self._adapters[task_name] = True
        print(f"[JnanaVerse] Adapter '{task_name}' added (Pfeiffer, r={reduction_factor}).")

    def enable_adapter(self, task_name: str):
        if task_name not in self._adapters:
            raise KeyError(f"Adapter '{task_name}' not registered. Call add_adapter() first.")
        self.base_model.set_active_adapters(task_name)
        print(f"[JnanaVerse] Adapter '{task_name}' is now active.")

    def disable_adapters(self):
        self.base_model.set_active_adapters(None)
        print("[JnanaVerse] All adapters disabled.")

    # ── Tokenisation utility ──────────────────────────────────
    def encode_text(
        self,
        text,
        max_length: int = 512,
        device: str = "cpu",
    ) -> dict:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return {k: v.to(device) for k, v in enc.items()}

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
