"""Zero-external-model local TF-IDF embedder for Turkish and English retrieval."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

_WORD_RE = re.compile(r"\w+", re.UNICODE)

_WORD_UNIGRAM_WEIGHT = 3.0
_WORD_BIGRAM_WEIGHT = 2.5
_CHAR_NGRAM_WEIGHT = 0.25
_STOPWORD_WEIGHT = 0.15

_STOPWORDS: frozenset[str] = frozenset({
    # Turkish function/question words that otherwise create noisy n-gram overlaps
    "acaba", "ama", "ancak", "artık", "aslında", "az", "bana", "bazı", "belki",
    "ben", "beni", "benim", "bir", "biraz", "birçok", "biri", "birkaç", "biz",
    "bize", "bizi", "bizim", "bu", "buna", "bunu", "bunun", "burada", "böyle",
    "bulunur", "çok", "çünkü", "da", "daha", "dahi", "de", "defa", "denir",
    "diğer", "diye", "edilir", "eğer", "en", "fakat", "gibi", "göre", "halen",
    "hangi", "hani", "hatta", "hem", "henüz", "hep", "hepsi", "her", "herkes",
    "hiç", "için", "ile", "ise", "işte", "kaç", "kadar", "kendi", "ki", "kim",
    "kime", "kimi", "kimse", "kullanılan", "kullanılır", "lakin", "mı", "mi",
    "mu", "mü", "nasıl", "ne", "neden", "nedir", "nerede", "nereden", "nereye",
    "neresi", "neresidir", "niçin", "niye", "o", "olan", "olarak", "oldu",
    "olduğu", "olur", "ona", "onu", "onun", "orada", "oysa", "öbür", "önce",
    "öyle", "sadece", "sahiptir", "sana", "sanki", "sen", "seni", "senin",
    "siz", "size", "sizi", "sizin", "sonra", "şayet", "şey", "şimdi", "şöyle",
    "şu", "şuna", "şunu", "tabii", "tüm", "üzere", "var", "ve", "veya",
    "verilir", "ya", "yani", "yapılır", "yine", "yok", "zaten",
    # English function/question words
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "may", "more",
    "most", "no", "not", "of", "on", "or", "our", "out", "over", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "to", "too", "under",
    "up", "use", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "why", "will", "with", "would", "you", "your",
})


def _normalize_text(text: str) -> str:
    """Apply Turkish-aware lowercasing without breaking English 'I' -> 'i'."""
    return text.replace("İ", "i").lower().replace("\u0307", "")


def _feature_weight(feature: str) -> float:
    """Return the weighting multiplier for a prefixed feature token."""
    if feature.startswith("sw:"):
        return _STOPWORD_WEIGHT
    if feature.startswith("w:"):
        return _WORD_UNIGRAM_WEIGHT
    if feature.startswith("b:"):
        return _WORD_BIGRAM_WEIGHT
    return _CHAR_NGRAM_WEIGHT


class LocalTfidfEmbedder:
    """Sublinear TF-IDF embedder using word unigrams, bigrams, and char n-grams."""

    def __init__(
        self, max_features: int = 4096, ngram_range: tuple[int, int] = (3, 4)
    ) -> None:
        if max_features <= 0:
            raise ValueError(f"max_features must be positive, got {max_features}")
        if (
            len(ngram_range) != 2
            or ngram_range[0] < 1
            or ngram_range[1] < ngram_range[0]
        ):
            raise ValueError(f"Invalid ngram_range: {ngram_range}")

        self.max_features = int(max_features)
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        self.vocab: dict[str, int] = {}
        self._idf: np.ndarray = np.empty(0, dtype=np.float32)
        self._weights: np.ndarray = np.empty(0, dtype=np.float32)
        self._fitted: bool = False

    def _extract_features(self, text: str) -> Counter[str]:
        """Extract word unigrams, bigrams, and boundary-padded char n-grams."""
        counts: Counter[str] = Counter()
        if not text:
            return counts

        normalized = _normalize_text(text)
        words = _WORD_RE.findall(normalized)
        if not words:
            return counts

        min_n, max_n = self.ngram_range
        content_words = [w for w in words if w not in _STOPWORDS]
        for i, word in enumerate(words):
            is_stop = word in _STOPWORDS
            counts[f"{'sw' if is_stop else 'w'}:{word}"] += 1

            # Skip char n-grams for stopwords and very short (<3 char) words like "su"/"ai"
            # so they don't produce noisy substring matches (e.g. "su" vs "sunar").
            if is_stop or len(word) < 3:
                continue

            padded = f"<{word}>"
            plen = len(padded)
            for n in range(min_n, max_n + 1):
                if plen < n:
                    continue
                for start in range(plen - n + 1):
                    counts[f"c:{padded[start : start + n]}"] += 1

        for i in range(len(content_words) - 1):
            counts[f"b:{content_words[i]} {content_words[i + 1]}"] += 1

        return counts

    @property
    def dim(self) -> int:
        return len(self.vocab)

    def fit(self, texts: list[str]) -> "LocalTfidfEmbedder":
        """Learn vocabulary and IDF weights from a corpus of documents."""
        if not texts:
            raise ValueError("Cannot fit LocalTfidfEmbedder on an empty text list.")

        n_docs = len(texts)
        df: Counter[str] = Counter()
        doc_features: list[Counter[str]] = []

        for text in texts:
            feats = self._extract_features(text)
            doc_features.append(feats)
            for feat in feats:
                df[feat] += 1

        if not df:
            raise ValueError("No valid word or character features found in texts.")

        idf_map: dict[str, float] = {
            feat: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0
            for feat, count in df.items()
        }

        corpus_importance: dict[str, float] = {feat: 0.0 for feat in df}
        for feats in doc_features:
            for feat, tf in feats.items():
                corpus_importance[feat] += (
                    _feature_weight(feat) * (1.0 + math.log(tf)) * idf_map[feat]
                )

        ranked_features = sorted(
            df.keys(),
            key=lambda feat: (-corpus_importance[feat], feat),
        )[: self.max_features]

        self.vocab = {feat: idx for idx, feat in enumerate(ranked_features)}
        self._idf = np.array(
            [idf_map[feat] for feat in ranked_features], dtype=np.float32
        )
        self._weights = np.array(
            [_feature_weight(feat) for feat in ranked_features], dtype=np.float32
        )
        self._fitted = True
        return self

    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """Encode texts into L2-normalized float32 embeddings compatible with FAISS IP."""
        _ = show_progress_bar
        _ = convert_to_numpy

        if not self._fitted:
            raise ValueError(
                "LocalTfidfEmbedder must be fitted before calling encode()."
            )

        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            feats = self._extract_features(text)
            for feat, tf in feats.items():
                col_idx = self.vocab.get(feat)
                if col_idx is not None:
                    embeddings[row_idx, col_idx] = (
                        self._weights[col_idx]
                        * (1.0 + math.log(tf))
                        * self._idf[col_idx]
                    )

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0.0
        embeddings[nonzero] /= norms[nonzero]
        return embeddings

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit on `texts` and return their normalized embeddings."""
        return self.fit(texts).encode(texts)

    def save(self, path: Path | str) -> None:
        """Persist fitted embedder vocabulary and IDF weights as UTF-8 JSON."""
        if not self._fitted:
            raise ValueError("Cannot save an unfitted LocalTfidfEmbedder.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "vocab": self.vocab,
            "idf": self._idf.tolist(),
            "weights": self._weights.tolist(),
            "fitted": self._fitted,
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path | str) -> "LocalTfidfEmbedder":
        """Load a persisted LocalTfidfEmbedder from a UTF-8 JSON file."""
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))

        embedder = cls(
            max_features=int(payload["max_features"]),
            ngram_range=(int(payload["ngram_range"][0]), int(payload["ngram_range"][1])),
        )
        embedder.vocab = {str(k): int(v) for k, v in payload["vocab"].items()}
        embedder._idf = np.array(payload["idf"], dtype=np.float32)
        embedder._weights = np.array(payload["weights"], dtype=np.float32)
        embedder._fitted = bool(payload.get("fitted", True))
        return embedder
