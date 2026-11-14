"""Zero-dependency Byte-Pair Encoding (BPE) tokenizer for Turkish and English text."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"
CTX_TOKEN = "<|context|>"
USR_TOKEN = "<|user|>"
AST_TOKEN = "<|assistant|>"

SPECIAL_TOKENS: list[str] = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    CTX_TOKEN,
    USR_TOKEN,
    AST_TOKEN,
]

_TURKISH_CHARS = "çğıöşüÇĞİÖŞÜ"
_WHITESPACE_BASE = ["\n", "\t"]
_ASCII_PRINTABLE = [chr(i) for i in range(32, 127)]

# Split on special tokens first (longest match first) so they remain atomic IDs.
_SPECIAL_SPLIT_RE = re.compile(
    "(" + "|".join(re.escape(tok) for tok in sorted(SPECIAL_TOKENS, key=len, reverse=True)) + ")"
)

# Partition non-special text losslessly into word chunks, space runs, single
# other whitespace chars, and individual punctuation/symbol chars.
_PRETOKEN_RE = re.compile(r"\w+| +|\s|[^\w\s]", re.UNICODE)


def format_chat_prompt(query: str) -> str:
    """Format a standalone user query into a chat prompt."""
    return f"{USR_TOKEN} {query.strip()} {AST_TOKEN} "


def format_rag_prompt(context: str, query: str) -> str:
    """Format retrieved context and user query into a RAG prompt."""
    return f"{CTX_TOKEN} {context.strip()} {USR_TOKEN} {query.strip()} {AST_TOKEN} "


def format_training_example(
    user: str, assistant: str, context: str | None = None
) -> tuple[str, str]:
    """Return (prompt, target) pair for supervised fine-tuning."""
    if context is not None and context.strip():
        prompt = format_rag_prompt(context, user)
    else:
        prompt = format_chat_prompt(user)
    target = f"{assistant.strip()}{EOS_TOKEN}"
    return prompt, target


def _merge_tuple(
    symbols: tuple[str, ...], pair: tuple[str, str], merged: str
) -> tuple[str, ...]:
    """Merge all non-overlapping left-to-right occurrences of `pair` in `symbols`."""
    out: list[str] = []
    i = 0
    n = len(symbols)
    a, b = pair
    while i < n:
        if i + 1 < n and symbols[i] == a and symbols[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    """Pure-Python Byte-Pair Encoding tokenizer with special chat/RAG tokens."""

    PAD_TOKEN = PAD_TOKEN
    BOS_TOKEN = BOS_TOKEN
    EOS_TOKEN = EOS_TOKEN
    UNK_TOKEN = UNK_TOKEN
    CTX_TOKEN = CTX_TOKEN
    USR_TOKEN = USR_TOKEN
    AST_TOKEN = AST_TOKEN
    SPECIAL_TOKENS = SPECIAL_TOKENS

    format_chat_prompt = staticmethod(format_chat_prompt)
    format_rag_prompt = staticmethod(format_rag_prompt)
    format_training_example = staticmethod(format_training_example)

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []
        self._merge_ranks: dict[tuple[str, str], int] = {}
        self._bpe_cache: dict[str, list[int]] = {}
        self._special_set: set[str] = set(SPECIAL_TOKENS)
        self._init_base_vocab()

    def _init_base_vocab(self, extra_chars: Iterable[str] = ()) -> None:
        """Initialize vocabulary with special tokens, base alphabet, and extra corpus chars."""
        self.vocab = {}
        self.id_to_token = {}
        self.merges = []
        self._merge_ranks = {}
        self._bpe_cache = {}

        for tok in SPECIAL_TOKENS:
            self._add_token(tok)

        for ch in _WHITESPACE_BASE + _ASCII_PRINTABLE + list(_TURKISH_CHARS):
            self._add_token(ch)

        for ch in sorted(set(extra_chars)):
            if ch not in self.vocab:
                self._add_token(ch)

    def _add_token(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        idx = len(self.vocab)
        self.vocab[token] = idx
        self.id_to_token[idx] = token
        return idx

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.vocab[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.vocab[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.vocab[UNK_TOKEN]

    @property
    def context_id(self) -> int:
        return self.vocab[CTX_TOKEN]

    @property
    def user_id(self) -> int:
        return self.vocab[USR_TOKEN]

    @property
    def assistant_id(self) -> int:
        return self.vocab[AST_TOKEN]

    def train(
        self, texts: list[str], vocab_size: int = 1200, min_frequency: int = 2
    ) -> None:
        """Train BPE merge rules from a list of text documents."""
        corpus_chars: set[str] = set()
        raw_word_counts: Counter[tuple[str, ...]] = Counter()

        for text in texts:
            if not text:
                continue
            for segment in _SPECIAL_SPLIT_RE.split(text):
                if not segment or segment in self._special_set:
                    continue
                corpus_chars.update(segment)
                for chunk in _PRETOKEN_RE.findall(segment):
                    raw_word_counts[tuple(chunk)] += 1

        self._init_base_vocab(extra_chars=corpus_chars)

        if vocab_size <= len(self.vocab) or not raw_word_counts:
            return

        words: list[tuple[str, ...]] = list(raw_word_counts.keys())
        freqs: list[int] = [raw_word_counts[w] for w in words]

        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        pair_to_words: dict[tuple[str, str], set[int]] = defaultdict(set)

        for idx, word in enumerate(words):
            freq = freqs[idx]
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq
                pair_to_words[pair].add(idx)

        while len(self.vocab) < vocab_size and pair_counts:
            best_pair, best_count = min(
                pair_counts.items(), key=lambda kv: (-kv[1], kv[0])
            )
            if best_count < min_frequency:
                break

            merged = best_pair[0] + best_pair[1]
            rank = len(self.merges)
            self.merges.append(best_pair)
            self._merge_ranks[best_pair] = rank
            self._add_token(merged)

            affected_indices = list(pair_to_words.get(best_pair, ()))
            for w_idx in affected_indices:
                old_word = words[w_idx]
                freq = freqs[w_idx]

                for i in range(len(old_word) - 1):
                    p = (old_word[i], old_word[i + 1])
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    word_set = pair_to_words.get(p)
                    if word_set is not None:
                        word_set.discard(w_idx)
                        if not word_set:
                            pair_to_words.pop(p, None)

                new_word = _merge_tuple(old_word, best_pair, merged)
                words[w_idx] = new_word

                for i in range(len(new_word) - 1):
                    p = (new_word[i], new_word[i + 1])
                    pair_counts[p] += freq
                    pair_to_words[p].add(w_idx)

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Apply learned BPE merges to a single pre-tokenized chunk."""
        cached = self._bpe_cache.get(chunk)
        if cached is not None:
            return cached

        symbols: list[str] = [ch if ch in self.vocab else UNK_TOKEN for ch in chunk]
        if len(symbols) > 1 and self._merge_ranks:
            while len(symbols) > 1:
                best_rank: int | None = None
                best_pair: tuple[str, str] | None = None
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    rank = self._merge_ranks.get(pair)
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank = rank
                        best_pair = pair
                if best_pair is None:
                    break
                merged = best_pair[0] + best_pair[1]
                symbols = list(_merge_tuple(tuple(symbols), best_pair, merged))

        ids = [self.vocab.get(sym, self.unk_id) for sym in symbols]
        if len(self._bpe_cache) < 50000:
            self._bpe_cache[chunk] = ids
        return ids

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        """Encode text into a list of integer token IDs."""
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)

        if text:
            for segment in _SPECIAL_SPLIT_RE.split(text):
                if not segment:
                    continue
                if segment in self._special_set:
                    ids.append(self.vocab[segment])
                else:
                    for chunk in _PRETOKEN_RE.findall(segment):
                        ids.extend(self._encode_chunk(chunk))

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a sequence of token IDs back into a string."""
        special_ids = {self.vocab[tok] for tok in SPECIAL_TOKENS}
        pieces: list[str] = []
        for idx in ids:
            if skip_special_tokens and idx in special_ids:
                continue
            tok = self.id_to_token.get(idx)
            if tok is None:
                if not skip_special_tokens:
                    pieces.append(UNK_TOKEN)
            else:
                pieces.append(tok)
        return "".join(pieces)

    def save(self, path: Path | str) -> None:
        """Persist tokenizer vocabulary and merge rules as UTF-8 JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "special_tokens": SPECIAL_TOKENS,
            "vocab": self.vocab,
            "merges": [[a, b] for a, b in self.merges],
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path | str) -> "BPETokenizer":
        """Load a previously saved BPETokenizer from a UTF-8 JSON file."""
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))

        tok = cls()
        tok.vocab = {str(k): int(v) for k, v in payload["vocab"].items()}
        tok.id_to_token = {int(v): str(k) for k, v in tok.vocab.items()}
        tok.merges = [(str(pair[0]), str(pair[1])) for pair in payload["merges"]]
        tok._merge_ranks = {pair: idx for idx, pair in enumerate(tok.merges)}
        tok._bpe_cache = {}
        return tok
