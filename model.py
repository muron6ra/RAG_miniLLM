"""Educational PyTorch Causal Transformer (MiniLLM) with weight tying and streaming."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiniLLMConfig:
    """Hyperparameter configuration for MiniLLM."""

    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {self.n_layer}")
        if self.n_head <= 0:
            raise ValueError(f"n_head must be positive, got {self.n_head}")
        if self.n_embd <= 0 or self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be positive and divisible by n_head ({self.n_head})"
            )


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with an explicit lower-triangular mask."""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        tril = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "bias",
            tril.view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, emb_dim = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(
            self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, emb_dim)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Pre-LayerNorm Transformer block."""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniLLM(nn.Module):
    """Decoder-only causal Transformer language model."""

    def __init__(self, config: MiniLLMConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie token embedding and output projection weights.
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=resid_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        """Return total number of unique model parameters (excluding buffers)."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logits and optional cross-entropy loss (ignoring target index -100)."""
        _, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}; "
                f"block_size is {self.config.block_size}."
            )

        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate_stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int | None = 20,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.15,
    ) -> Iterator[int]:
        """Yield generated token IDs one by one for a batch-size-1 prompt tensor."""
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        if idx.dim() != 2 or idx.size(0) != 1:
            raise ValueError(
                f"generate_stream requires shape [1, T], got {tuple(idx.shape)}"
            )
        if idx.size(1) == 0:
            raise ValueError("Input sequence idx must not be empty.")
        if max_new_tokens <= 0:
            return

        was_training = self.training
        self.eval()
        current = idx
        prompt_len = idx.size(1)

        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    idx_cond = current[:, -self.config.block_size :]
                    logits, _ = self(idx_cond)
                    next_logits = logits[:, -1, :].clone()

                    if (
                        repetition_penalty is not None
                        and repetition_penalty != 1.0
                        and current.size(1) > prompt_len
                    ):
                        # Penalize only tokens generated so far so RAG context words aren't suppressed.
                        seen_ids = torch.unique(current[0, prompt_len:])
                        seen_logits = next_logits[0, seen_ids]
                        next_logits[0, seen_ids] = torch.where(
                            seen_logits > 0,
                            seen_logits / repetition_penalty,
                            seen_logits * repetition_penalty,
                        )

                    if temperature <= 0.0:
                        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                    else:
                        scaled = next_logits / temperature
                        if top_k is not None and top_k > 0:
                            k = min(top_k, scaled.size(-1))
                            top_vals, _ = torch.topk(scaled, k)
                            scaled[scaled < top_vals[:, [-1]]] = float("-inf")
                        probs = F.softmax(scaled, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                    next_id = int(next_token.item())
                    if eos_token_id is not None and next_id == eos_token_id:
                        break

                    current = torch.cat((current, next_token), dim=1)
                    yield next_id
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int | None = 20,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.15,
    ) -> torch.Tensor:
        """Generate up to `max_new_tokens` and return full [1, T + generated] tensor."""
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)

        generated = list(
            self.generate_stream(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
            )
        )
        if not generated:
            return idx

        gen_tensor = torch.tensor([generated], dtype=idx.dtype, device=idx.device)
        return torch.cat((idx, gen_tensor), dim=1)

    def save_checkpoint(self, path: Path | str, meta: dict | None = None) -> None:
        """Save model config, weights, and optional metadata dictionary."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "config": asdict(self.config),
            "model_state_dict": self.state_dict(),
            "meta": meta or {},
        }
        torch.save(checkpoint, target)

    @classmethod
    def load_checkpoint(
        cls, path: Path | str, device: str = "cpu"
    ) -> tuple["MiniLLM", dict]:
        """Load checkpoint safely onto `device`, switch to eval mode, and return (model, meta)."""
        source = Path(path)
        checkpoint = torch.load(source, map_location=device, weights_only=True)
        config = MiniLLMConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        meta = checkpoint.get("meta", {})
        return model, meta
