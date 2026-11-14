"""Zero-external-model PyTorch training script for Turkish Chat & RAG Mini-LLM."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from model import MiniLLM, MiniLLMConfig
from tokenizer import BPETokenizer, format_training_example


@dataclass(frozen=True)
class Example:
    """Single supervised conversation or RAG Q&A record."""

    user: str
    assistant: str
    context: str | None = None
    source: str = ""

    @property
    def is_everyday(self) -> bool:
        return self.context is None or not self.context.strip()


EncodedSample = tuple[list[int], list[int], list[float]]


class EncodedDataset(Dataset):
    """In-memory dataset of pre-tokenized (input_ids, target_ids, loss_weights) tensors."""

    def __init__(self, samples: list[EncodedSample]) -> None:
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [
            (
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(w, dtype=torch.float32),
            )
            for x, y, w in samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Group similar-length sequences into batches with epoch-wise jitter and shuffling."""

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        jitter: float = 8.0,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.jitter = max(0.0, float(jitter))
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self._epoch = 0

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        n = len(self.lengths)
        if n == 0:
            return

        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        if self.shuffle:
            indices = sorted(
                range(n),
                key=lambda i: (
                    self.lengths[i]
                    + (rng.uniform(-self.jitter, self.jitter) if self.jitter > 0.0 else 0.0),
                    rng.random(),
                ),
            )
        else:
            indices = sorted(range(n), key=lambda i: self.lengths[i])

        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, n, self.batch_size)
        ]
        if self.shuffle:
            rng.shuffle(batches)

        for batch in batches:
            yield batch


def resolve_device(device_arg: str) -> str:
    """Resolve target PyTorch device ('auto', 'cuda', or 'cpu')."""
    choice = device_arg.lower().strip()
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested via --device cuda, but torch.cuda.is_available() is False."
            )
        return "cuda"
    if choice == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported --device value: {device_arg!r}")


def configure_runtime(seed: int, device: str) -> None:
    """Configure UTF-8 console output, deterministic seeds, and device backend."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda or "unknown"
        print(
            f"CUDA verified: torch={torch.__version__} | cuda={cuda_ver} | "
            f"GPU[0]={gpu_name} (count={torch.cuda.device_count()})"
        )
    else:
        num_threads = max(1, min(os.cpu_count() or 4, 8))
        torch.set_num_threads(num_threads)
        print(f"CPU runtime: torch={torch.__version__} | threads={num_threads}")


def load_jsonl_datasets(dataset_dir: Path) -> list[Example]:
    """Load all sorted *.jsonl files from `dataset_dir`."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    files = sorted(dataset_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in: {dataset_dir}")

    examples: list[Example] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                user = str(record["user"]).strip()
                assistant = str(record["assistant"]).strip()
                raw_ctx = record.get("context")
                context = str(raw_ctx).strip() if raw_ctx is not None else None
                if context == "":
                    context = None
                if not user or not assistant:
                    raise ValueError(
                        f"Empty user/assistant in {path.name}:{line_no}"
                    )
                examples.append(
                    Example(
                        user=user,
                        assistant=assistant,
                        context=context,
                        source=path.name,
                    )
                )
    return examples


def load_knowledge_texts(data_dir: Path) -> list[str]:
    """Load optional *.txt knowledge documents for BPE tokenizer coverage and pretraining."""
    if not data_dir.is_dir():
        return []
    texts: list[str] = []
    for path in sorted(data_dir.glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            texts.append(content)
    return texts


def split_train_val(
    examples: list[Example], val_split: float, seed: int
) -> tuple[list[Example], list[Example]]:
    """Deterministically split examples into train and validation sets.

    Stratifies by everyday vs. RAG context so both domains appear in validation.
    """
    if not (0.0 <= val_split < 1.0):
        raise ValueError(f"val_split must be in [0.0, 1.0), got {val_split}")

    if val_split == 0.0 or len(examples) < 2:
        return list(examples), []

    rng = random.Random(seed)
    everyday = [ex for ex in examples if ex.is_everyday]
    rag = [ex for ex in examples if not ex.is_everyday]

    train_examples: list[Example] = []
    val_examples: list[Example] = []

    for group in (everyday, rag):
        if not group:
            continue
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * val_split))
        if len(shuffled) > 1:
            n_val = min(n_val, len(shuffled) - 1)
        else:
            n_val = 0
        val_examples.extend(shuffled[:n_val])
        train_examples.extend(shuffled[n_val:])

    if not val_examples and len(train_examples) > 1:
        val_examples.append(train_examples.pop())

    rng.shuffle(train_examples)
    rng.shuffle(val_examples)
    return train_examples, val_examples


def turkish_capitalize(text: str) -> str:
    """Capitalize the first character using Turkish casing rules (i->İ, ı->I)."""
    if not text:
        return text
    first = text[0]
    if first == "i":
        head = "İ"
    elif first == "ı":
        head = "I"
    else:
        head = first.upper()
    return head + text[1:]


def turkish_lower(text: str) -> str:
    """Lowercase text handling Turkish 'İ' -> 'i' cleanly."""
    return text.replace("İ", "i").lower().replace("\u0307", "")


_QUESTION_WORD_RE = re.compile(
    r"(?<!\w)(?:mı|mi|mu|mü|mısın|misin|musun|müsün|mısınız|misiniz|"
    r"kaç|kaçtır|kaçı|kacı|ne|nedir|neler|nelerdir|nasıl|nasılsın|"
    r"nasılsınız|naber|nbr|kim|kimsin|kimdir|neresi|neresidir|nerede|"
    r"nereden|nereye|hangi|hangisi|hangileridir|neden|niçin|niye|acaba|"
    r"what|how|who|when|where|which|why)(?!\w)",
    re.IGNORECASE,
)


def _looks_like_question(text: str) -> bool:
    """Return True if an everyday prompt is phrased as a question without trailing punctuation."""
    stripped = text.strip()
    if not stripped or stripped.endswith(("?", "!", ".")):
        return False
    return bool(_QUESTION_WORD_RE.search(turkish_lower(stripped)))


def augment_everyday_train(
    train_examples: list[Example], val_examples: list[Example]
) -> list[Example]:
    """Add capitalized, '?' -stripped, and '?' -appended everyday variants in train only.

    Never duplicates existing prompts or leaks any validation user prompt.
    """
    blocked_users: set[str] = {
        ex.user.strip() for ex in val_examples if ex.is_everyday
    } | {ex.user.strip() for ex in train_examples if ex.is_everyday}

    augmented: list[Example] = list(train_examples)
    for ex in train_examples:
        if not ex.is_everyday:
            continue

        candidates: list[str] = []
        cap = turkish_capitalize(ex.user)
        candidates.append(cap)

        if ex.user.endswith("?"):
            stripped = ex.user[:-1].strip()
            if stripped:
                candidates.append(stripped)
                candidates.append(turkish_capitalize(stripped))
        elif _looks_like_question(ex.user):
            candidates.append(f"{ex.user}?")
            candidates.append(f"{cap}?")

        for variant in candidates:
            if variant in blocked_users:
                continue
            blocked_users.add(variant)
            augmented.append(
                Example(
                    user=variant,
                    assistant=ex.assistant,
                    context=None,
                    source=ex.source,
                )
            )
    return augmented


def augment_rag_train(
    train_examples: list[Example], val_examples: list[Example]
) -> list[Example]:
    """Add lowercase and '?' -stripped query variants for RAG examples in train only."""
    blocked_pairs: set[tuple[str, str]] = {
        ((ex.context or "").strip(), ex.user.strip())
        for ex in val_examples
        if not ex.is_everyday
    } | {
        ((ex.context or "").strip(), ex.user.strip())
        for ex in train_examples
        if not ex.is_everyday
    }

    augmented: list[Example] = list(train_examples)
    for ex in train_examples:
        if ex.is_everyday:
            continue

        ctx_key = (ex.context or "").strip()
        candidates: list[str] = []
        lowered = turkish_lower(ex.user).strip()
        if lowered:
            candidates.append(lowered)

        if ex.user.endswith("?"):
            stripped = ex.user[:-1].strip()
            if stripped:
                candidates.append(stripped)
                stripped_lower = turkish_lower(stripped).strip()
                if stripped_lower:
                    candidates.append(stripped_lower)

        for variant in candidates:
            key = (ctx_key, variant)
            if key in blocked_pairs:
                continue
            blocked_pairs.add(key)
            augmented.append(
                Example(
                    user=variant,
                    assistant=ex.assistant,
                    context=ex.context,
                    source=ex.source,
                )
            )
    return augmented


def upsample_rag_examples(
    train_examples: list[Example], rag_upsample: int
) -> list[Example]:
    """Repeat RAG examples in the training split so they aren't drowned by chat."""
    if rag_upsample <= 1:
        return list(train_examples)

    out: list[Example] = list(train_examples)
    rag_only = [ex for ex in train_examples if not ex.is_everyday]
    for _ in range(rag_upsample - 1):
        out.extend(rag_only)
    return out


def augment_rag_distractors(
    train_examples: list[Example],
    val_examples: list[Example],
    copies: int = 1,
    seed: int = 42,
    max_combined_chars: int = 1550,
) -> list[Example]:
    """Synthesize multi-chunk RAG contexts (`C --- D` / `D --- C`) from train chunks only.

    Teaches selective attention across `---` chunk boundaries when FAISS returns 2 chunks.
    Skips negative ('yer almıyor') records so a distractor never invalidates a negative label.
    """
    if copies <= 0:
        return list(train_examples)

    chunk_pool = sorted(
        {
            ex.context.strip()
            for ex in train_examples
            if not ex.is_everyday and ex.context and "---" not in ex.context
        }
    )
    if len(chunk_pool) < 2:
        return list(train_examples)

    blocked_pairs: set[tuple[str, str]] = {
        ((ex.context or "").strip(), ex.user.strip())
        for ex in (train_examples + val_examples)
        if not ex.is_everyday
    }

    rng = random.Random(seed + 7919)
    augmented: list[Example] = list(train_examples)
    pos_idx = 0

    for ex in train_examples:
        if ex.is_everyday or not ex.context or "---" in ex.context:
            continue
        if "yer almıyor" in ex.assistant.lower():
            continue

        base_ctx = ex.context.strip()
        candidates = [
            cand
            for cand in chunk_pool
            if cand != base_ctx
            and len(base_ctx) + len(cand) + 7 <= max_combined_chars
        ]
        if not candidates:
            continue

        rng.shuffle(candidates)
        added_here = 0
        for cand in candidates:
            if added_here >= copies:
                break
            place_after = (pos_idx + added_here) % 2 == 0
            combined_ctx = (
                f"{base_ctx}\n\n---\n\n{cand}"
                if place_after
                else f"{cand}\n\n---\n\n{base_ctx}"
            )
            key = (combined_ctx, ex.user.strip())
            if key in blocked_pairs:
                continue
            blocked_pairs.add(key)
            augmented.append(
                Example(
                    user=ex.user,
                    assistant=ex.assistant,
                    context=combined_ctx,
                    source=ex.source,
                )
            )
            added_here += 1
        pos_idx += 1

    return augmented


def build_tokenizer_corpus(
    train_examples: list[Example], knowledge_texts: list[str]
) -> list[str]:
    """Collect training-only texts, capitalized prompt variants, and knowledge docs for BPE."""
    corpus: list[str] = []
    for ex in train_examples:
        if ex.context:
            corpus.append(ex.context)
        corpus.append(ex.user)
        cap_user = turkish_capitalize(ex.user)
        if cap_user != ex.user:
            corpus.append(cap_user)
        corpus.append(ex.assistant)
    corpus.extend(knowledge_texts)
    return corpus


def encode_example(
    example: Example,
    tokenizer: BPETokenizer,
    block_size: int,
    mask_prompt: bool = True,
    prompt_loss_weight: float = 0.0,
) -> EncodedSample:
    """Encode an example into causal LM (x, y, weights) sequences within `block_size`."""
    prompt_str, target_str = format_training_example(
        example.user, example.assistant, example.context
    )
    prompt_ids = tokenizer.encode(prompt_str)
    target_ids = tokenizer.encode(target_str)

    max_full_len = block_size + 1
    if len(target_ids) >= max_full_len:
        prompt_ids = prompt_ids[-1:]
        target_ids = target_ids[: max_full_len - 2] + [tokenizer.eos_id]
    elif len(prompt_ids) + len(target_ids) > max_full_len:
        max_prompt_len = max_full_len - len(target_ids)
        if (
            prompt_ids
            and prompt_ids[0] == tokenizer.context_id
            and tokenizer.user_id in prompt_ids
        ):
            user_pos = prompt_ids.index(tokenizer.user_id)
            tail_ids = prompt_ids[user_pos:]
            if 1 + len(tail_ids) <= max_prompt_len:
                ctx_budget = max_prompt_len - 1 - len(tail_ids)
                prompt_ids = (
                    [prompt_ids[0]] + prompt_ids[1 : 1 + ctx_budget] + tail_ids
                )
            else:
                prompt_ids = prompt_ids[-max_prompt_len:]
        else:
            prompt_ids = prompt_ids[-max_prompt_len:]

    full_ids = prompt_ids + target_ids
    prompt_len = len(prompt_ids)

    x = full_ids[:-1]
    y = full_ids[1:]
    weights = [1.0] * len(y)

    # x[prompt_len - 1] is the final prompt token predicting y[prompt_len - 1] (= target_ids[0]).
    if mask_prompt and prompt_len > 1:
        n_prompt_targets = prompt_len - 1
        if prompt_loss_weight > 0.0:
            weights[:n_prompt_targets] = [float(prompt_loss_weight)] * n_prompt_targets
        else:
            y[:n_prompt_targets] = [-100] * n_prompt_targets
            weights[:n_prompt_targets] = [0.0] * n_prompt_targets

    return x, y, weights


def build_pretrain_samples(
    knowledge_texts: list[str],
    train_examples: list[Example],
    tokenizer: BPETokenizer,
    block_size: int,
) -> list[EncodedSample]:
    """Build pure LM pretraining chunks from knowledge docs and unique training contexts."""
    raw_texts: list[str] = []
    for doc in knowledge_texts:
        for para in doc.split("\n\n"):
            p = para.strip()
            if p:
                raw_texts.append(p)
        raw_texts.append(doc.strip())

    seen_contexts: set[str] = set()
    for ex in train_examples:
        if ex.context and "---" not in ex.context and ex.context not in seen_contexts:
            seen_contexts.add(ex.context)
            raw_texts.append(ex.context)

    samples: list[EncodedSample] = []
    stride = max(32, block_size // 2)
    for text in raw_texts:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(ids) < 2:
            continue
        if len(ids) <= block_size + 1:
            samples.append((ids[:-1], ids[1:], [1.0] * (len(ids) - 1)))
        else:
            for start in range(0, len(ids) - 1, stride):
                window = ids[start : start + block_size + 1]
                if len(window) < 8:
                    break
                samples.append((window[:-1], window[1:], [1.0] * (len(window) - 1)))
                if start + block_size + 1 >= len(ids):
                    break
    return samples


def make_collate_fn(pad_id: int):
    """Create a batch collation function padding x with `pad_id`, y with -100, and w with 0.0."""

    def collate_batch(
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max(x.size(0) for x, _, _ in batch)
        bsz = len(batch)
        x_tensor = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        y_tensor = torch.full((bsz, max_len), -100, dtype=torch.long)
        w_tensor = torch.zeros((bsz, max_len), dtype=torch.float32)
        for row_idx, (x_seq, y_seq, w_seq) in enumerate(batch):
            seq_len = x_seq.size(0)
            x_tensor[row_idx, :seq_len] = x_seq
            y_tensor[row_idx, :seq_len] = y_seq
            w_tensor[row_idx, :seq_len] = w_seq
        return x_tensor, y_tensor, w_tensor

    return collate_batch


def build_optimizer(
    model: MiniLLM, lr: float, weight_decay: float
) -> torch.optim.AdamW:
    """Configure AdamW with weight decay only on 2D+ parameters (not LayerNorm gains)."""
    decay_params: list[torch.nn.Parameter] = []
    nodecay_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay down to `min_lr_ratio * base_lr`."""
    total = max(1, int(total_epochs))
    warmup = max(0, min(int(warmup_epochs), total - 1))

    def lr_lambda(epoch_idx: int) -> float:
        step = epoch_idx + 1
        if warmup > 0 and step <= warmup:
            return max(1e-3, step / warmup)
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def compute_weighted_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted (optionally label-smoothed) cross-entropy loss and weight sum."""
    per_token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-100,
        reduction="none",
        label_smoothing=label_smoothing,
    ).view_as(targets)
    weight_sum = weights.sum()
    loss = (per_token_loss * weights).sum() / weight_sum.clamp_min(1e-8)
    return loss, weight_sum


def train_one_epoch(
    model: MiniLLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.0,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Run one training epoch and return weight-averaged cross-entropy loss."""
    model.train()
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_weight = torch.zeros((), dtype=torch.float64, device=device)

    amp_dtype = (
        torch.bfloat16
        if (use_amp and device == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    for x, y, w in loader:
        if float(w.sum()) <= 0.0:
            continue

        x = x.to(device)
        y = y.to(device)
        w = w.to(device)

        optimizer.zero_grad(set_to_none=True)
        ctx = (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if (use_amp and device == "cuda")
            else nullcontext()
        )
        with ctx:
            logits, _ = model(x)
            loss, weight_sum = compute_weighted_ce_loss(
                logits, y, w, label_smoothing=label_smoothing
            )

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.detach().to(torch.float64) * weight_sum.to(torch.float64)
        total_weight += weight_sum.to(torch.float64)

    denom = float(total_weight.item())
    if denom <= 0.0:
        return 0.0
    return float(total_loss.item()) / denom


@torch.no_grad()
def evaluate(
    model: MiniLLM,
    loader: DataLoader | None,
    device: str,
    use_amp: bool = False,
) -> float | None:
    """Evaluate unsmoothed cross-entropy loss on validation loader."""
    if loader is None or len(loader.dataset) == 0:
        return None

    model.eval()
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_weight = torch.zeros((), dtype=torch.float64, device=device)

    amp_dtype = (
        torch.bfloat16
        if (use_amp and device == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype)
        if (use_amp and device == "cuda")
        else nullcontext()
    )

    for x, y, w in loader:
        if float(w.sum()) <= 0.0:
            continue

        x = x.to(device)
        y = y.to(device)
        w = w.to(device)

        with ctx:
            logits, _ = model(x)
            loss, weight_sum = compute_weighted_ce_loss(
                logits, y, w, label_smoothing=0.0
            )
        total_loss += loss.to(torch.float64) * weight_sum.to(torch.float64)
        total_weight += weight_sum.to(torch.float64)

    denom = float(total_weight.item())
    if denom <= 0.0:
        return 0.0
    return float(total_loss.item()) / denom


def make_loader(
    samples: list[EncodedSample],
    pad_id: int,
    batch_size: int,
    bucket_jitter: float,
    seed: int,
    shuffle: bool,
    device: str,
) -> DataLoader:
    """Build a length-bucketed DataLoader minimizing sequence padding waste."""
    _ = device
    lengths = [len(x) for x, _, _ in samples]
    batch_sampler = LengthBucketBatchSampler(
        lengths=lengths,
        batch_size=batch_size,
        jitter=bucket_jitter,
        seed=seed,
        shuffle=shuffle,
    )
    return DataLoader(
        EncodedDataset(samples),
        batch_sampler=batch_sampler,
        collate_fn=make_collate_fn(pad_id),
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for tokenizer and MiniLLM training."""
    parser = argparse.ArgumentParser(
        description="Train zero-external-model Turkish Chat & RAG Mini-LLM."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("./dataset"))
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--out-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device: auto (prefer CUDA), cuda, or cpu.",
    )
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=576)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-mask-prompt",
        action="store_true",
        help="Compute loss uniformly on prompt tokens as well instead of masking prompt tokens.",
    )
    parser.add_argument(
        "--prompt-loss-weight",
        type=float,
        default=0.05,
        help="Relative loss weight on prompt tokens when prompt masking is active (default: 0.05).",
    )
    parser.add_argument(
        "--rag-upsample",
        type=int,
        default=1,
        help="Repeat factor for RAG training examples (training split only, default: 1).",
    )
    parser.add_argument(
        "--rag-distractor-aug",
        type=int,
        default=1,
        help="Synthesize N multi-chunk distractor contexts per positive RAG train example (default: 1).",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.01,
        help="Label smoothing factor for training cross-entropy loss (default: 0.01).",
    )
    parser.add_argument(
        "--bucket-jitter",
        type=float,
        default=8.0,
        help="Length jitter in tokens for length-bucketed batch sampling (default: 8.0).",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=25,
        help="Optional stage-1 LM pretraining epochs on knowledge docs + contexts (default: 25).",
    )
    parser.add_argument(
        "--pretrain-lr",
        type=float,
        default=2e-3,
        help="Learning rate for optional stage-1 LM pretraining (default: 2e-3).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed-precision autocast on CUDA (default: exact float32).",
    )
    parser.add_argument(
        "--augment-case",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add capitalized/lowercase and '?' -stripped query variants in train split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    configure_runtime(args.seed, device)
    mask_prompt = not args.no_mask_prompt
    prompt_loss_weight = 1.0 if not mask_prompt else max(0.0, float(args.prompt_loss_weight))
    use_amp = bool(args.amp and device == "cuda")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_examples = load_jsonl_datasets(args.dataset_dir)
    knowledge_texts = load_knowledge_texts(args.data_dir)
    raw_train, val_examples = split_train_val(
        raw_examples, args.val_split, args.seed
    )

    train_examples = augment_rag_distractors(
        raw_train,
        val_examples,
        copies=args.rag_distractor_aug,
        seed=args.seed,
    )
    if args.augment_case:
        train_examples = augment_everyday_train(train_examples, val_examples)
        train_examples = augment_rag_train(train_examples, val_examples)

    n_before_upsample = len(train_examples)
    train_examples = upsample_rag_examples(train_examples, args.rag_upsample)

    tokenizer = BPETokenizer()
    # Fit BPE on balanced raw_train + knowledge_texts before RAG upsampling distorts counts.
    bpe_corpus = build_tokenizer_corpus(raw_train, knowledge_texts)
    tokenizer.train(
        bpe_corpus, vocab_size=args.vocab_size, min_frequency=args.min_freq
    )
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    train_samples = [
        encode_example(
            ex,
            tokenizer,
            args.block_size,
            mask_prompt=mask_prompt,
            prompt_loss_weight=prompt_loss_weight,
        )
        for ex in train_examples
    ]
    # Validate strictly on assistant answer tokens (or full sequence if --no-mask-prompt).
    val_samples = [
        encode_example(
            ex,
            tokenizer,
            args.block_size,
            mask_prompt=mask_prompt,
            prompt_loss_weight=0.0,
        )
        for ex in val_examples
    ]

    train_loader = make_loader(
        train_samples,
        pad_id=tokenizer.pad_id,
        batch_size=args.batch_size,
        bucket_jitter=args.bucket_jitter,
        seed=args.seed,
        shuffle=True,
        device=device,
    )
    val_loader = (
        make_loader(
            val_samples,
            pad_id=tokenizer.pad_id,
            batch_size=args.batch_size,
            bucket_jitter=0.0,
            seed=args.seed,
            shuffle=False,
            device=device,
        )
        if val_samples
        else None
    )

    config = MiniLLMConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = MiniLLM(config).to(device)
    param_count = model.count_parameters()

    n_everyday_train = sum(1 for ex in train_examples if ex.is_everyday)
    n_rag_train = len(train_examples) - n_everyday_train
    avg_len = sum(len(x) for x, _, _ in train_samples) / max(1, len(train_samples))
    max_len = max((len(x) for x, _, _ in train_samples), default=0)

    print(
        f"Device: {device} | Raw examples: {len(raw_examples)} "
        f"(train={len(raw_train)} -> {n_before_upsample} after aug={args.augment_case} -> "
        f"{len(train_examples)} after rag_upsample={args.rag_upsample} "
        f"[chat={n_everyday_train}, rag={n_rag_train}], val={len(val_examples)})"
    )
    print(
        f"Tokenizer vocab_size: {tokenizer.vocab_size} (merges={len(tokenizer.merges)}) | "
        f"Model params: {param_count:,} | block_size={args.block_size} (train avg={avg_len:.1f}, max={max_len})"
    )
    print(
        f"Objective: mask_prompt={mask_prompt}, prompt_loss_weight={prompt_loss_weight:.2f}, "
        f"label_smoothing={args.label_smoothing:.3f}, amp={use_amp}"
    )

    scaler = (
        torch.amp.GradScaler("cuda", enabled=(use_amp and not torch.cuda.is_bf16_supported()))
        if device == "cuda"
        else None
    )

    start_time = time.perf_counter()

    if args.pretrain_epochs > 0:
        pretrain_samples = build_pretrain_samples(
            knowledge_texts, train_examples, tokenizer, args.block_size
        )
        if pretrain_samples:
            pretrain_loader = make_loader(
                pretrain_samples,
                pad_id=tokenizer.pad_id,
                batch_size=args.batch_size,
                bucket_jitter=args.bucket_jitter,
                seed=args.seed + 1000,
                shuffle=True,
                device=device,
            )
            pt_opt = build_optimizer(model, lr=args.pretrain_lr, weight_decay=args.weight_decay)
            pt_sched = build_lr_scheduler(
                pt_opt, total_epochs=args.pretrain_epochs, warmup_epochs=min(3, args.pretrain_epochs // 3)
            )
            print(
                f"Stage 1 LM Pretraining: {args.pretrain_epochs} epochs on {len(pretrain_samples)} chunks..."
            )
            for ep in range(1, args.pretrain_epochs + 1):
                pt_loss = train_one_epoch(
                    model,
                    pretrain_loader,
                    pt_opt,
                    device,
                    label_smoothing=args.label_smoothing,
                    use_amp=use_amp,
                    scaler=scaler,
                )
                pt_sched.step()
                if ep == 1 or ep % 5 == 0 or ep == args.pretrain_epochs:
                    print(f"  Pretrain {ep:3d}/{args.pretrain_epochs} | loss={pt_loss:.4f}")

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(
        optimizer, total_epochs=args.epochs, warmup_epochs=args.warmup_epochs
    )

    checkpoint_path = out_dir / "mini_llm.pt"
    best_metric = float("inf")
    best_epoch = 0
    best_train_loss = float("inf")
    best_val_loss: float | None = None
    history: list[dict[str, float | int | None]] = []

    final_train_loss = float("inf")
    final_val_loss: float | None = None

    for epoch in range(1, args.epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            label_smoothing=args.label_smoothing,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_loss = evaluate(model, val_loader, device, use_amp=use_amp)
        scheduler.step()

        final_train_loss = train_loss
        final_val_loss = val_loss
        monitor = val_loss if val_loss is not None else train_loss

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6) if val_loss is not None else None,
                "lr": current_lr,
            }
        )

        improved = monitor < best_metric
        if improved:
            best_metric = monitor
            best_epoch = epoch
            best_train_loss = train_loss
            best_val_loss = val_loss
            meta = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss) if val_loss is not None else None,
                "num_examples": int(len(train_examples)),
                "raw_examples": int(len(raw_examples)),
                "num_val_examples": int(len(val_examples)),
                "vocab_size": int(tokenizer.vocab_size),
                "params": int(param_count),
                "seed": int(args.seed),
                "device": device,
                "mask_prompt": bool(mask_prompt),
                "prompt_loss_weight": float(prompt_loss_weight),
                "rag_upsample": int(args.rag_upsample),
                "rag_distractor_aug": int(args.rag_distractor_aug),
                "label_smoothing": float(args.label_smoothing),
                "augment_case": bool(args.augment_case),
            }
            model.save_checkpoint(checkpoint_path, meta=meta)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs or improved:
            val_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
            mark = " *" if improved else ""
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_str} | "
                f"lr={current_lr:.6f}{mark}"
            )

    elapsed = time.perf_counter() - start_time
    metrics_payload = {
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "final_epoch": int(args.epochs),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss) if final_val_loss is not None else None,
        "num_examples": int(len(train_examples)),
        "raw_examples": int(len(raw_examples)),
        "raw_train_examples": int(len(raw_train)),
        "num_val_examples": int(len(val_examples)),
        "vocab_size": int(tokenizer.vocab_size),
        "params": int(param_count),
        "elapsed_seconds": round(elapsed, 2),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "config": {
            "block_size": args.block_size,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_split": args.val_split,
            "seed": args.seed,
            "mask_prompt": mask_prompt,
            "prompt_loss_weight": prompt_loss_weight,
            "rag_upsample": args.rag_upsample,
            "rag_distractor_aug": args.rag_distractor_aug,
            "label_smoothing": args.label_smoothing,
            "bucket_jitter": args.bucket_jitter,
            "pretrain_epochs": args.pretrain_epochs,
            "augment_case": args.augment_case,
            "amp": use_amp,
        },
        "history": history,
    }
    metrics_path = out_dir / "training_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_val_str = f"{best_val_loss:.4f}" if best_val_loss is not None else "n/a"
    print(
        f"\nTraining complete in {elapsed:.2f}s. "
        f"Best epoch={best_epoch} (train_loss={best_train_loss:.4f}, val_loss={best_val_str})."
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved tokenizer:  {tokenizer_path}")
    print(f"Saved metrics:    {metrics_path}")


if __name__ == "__main__":
    main()
