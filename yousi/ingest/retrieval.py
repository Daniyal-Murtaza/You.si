"""Local embeddings and hybrid retrieval over the vault.

Embeddings run on the person's own machine with an open model (bge-small,
384 dims, ~130 MB, downloaded once). No text leaves the computer for
retrieval. Search combines BM25 (exact words: names, places, dates) with
vector similarity (meaning), which is what works for a few thousand short,
personal memories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from yousi.logging_setup import get_logger
from yousi.vault import Memory, Vault

log = get_logger("ingest.retrieval")


class Embedder(Protocol):
    dims: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class LocalEmbedder:
    """sentence-transformers model, loaded lazily (first load takes ~30s)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._name = model_name
        self._model = None
        self.dims = 384

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading embedding model %s (first time is slow)", self._name)
            self._model = SentenceTransformer(self._name)
            self.dims = int(self._model.get_sentence_embedding_dimension() or 384)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        vecs = self._load().encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


class HashEmbedder:
    """Deterministic, dependency-free embedder for tests. Bag of words, not semantic.

    Uses a stable hash: Python's ``hash()`` is salted per process, which would
    make tests flaky.
    """

    def __init__(self, dims: int = 64) -> None:
        self.dims = dims

    def encode(self, texts: list[str]) -> np.ndarray:
        import zlib

        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                out[i, zlib.crc32(tok.encode()) % self.dims] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


_TOKEN = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "from",
        "at",
        "as",
        "is",
        "are",
        "be",
        "was",
        "were",
        "i",
        "my",
        "me",
        "you",
        "your",
        "it",
        "its",
        "we",
        "our",
        "they",
        "their",
        "this",
        "that",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "not",
        "no",
    }
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True)
class Hit:
    memory: Memory
    score: float
    why: str  # "both" | "words" | "meaning"


class Retriever:
    """Hybrid search; rebuilt cheaply whenever the vault changes."""

    def __init__(self, vault: Vault, embedder: Embedder) -> None:
        self._vault = vault
        self._embedder = embedder
        self._memories: list[Memory] = []
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._stamp = -1

    def refresh(self) -> None:
        stamp = self._vault.count()
        if stamp == self._stamp and self._memories:
            return
        self._memories = self._vault.all_memories()
        corpus = [
            tokenize(f"{m.domain} {m.kind} {' '.join(m.tags)} {m.text}") for m in self._memories
        ]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._ids, self._matrix = self._vault.embeddings()
        self._stamp = stamp

    def search(self, query: str, *, top_k: int = 8, domain: str | None = None) -> list[Hit]:
        self.refresh()
        if not self._memories:
            return []
        by_id = {m.memory_id: m for m in self._memories}
        scores: dict[str, tuple[float, set[str]]] = {}

        if self._bm25 is not None:
            tokens = tokenize(query)
            if tokens:
                raw = self._bm25.get_scores(tokens)
                top = max(float(raw.max()), 1e-9)
                for m, s in zip(self._memories, raw, strict=True):
                    if s > 0:
                        scores[m.memory_id] = (float(s) / top, {"words"})

        if self._ids and self._matrix.size:
            q = self._embedder.encode([query])[0]
            sims = self._matrix @ q
            order = np.argsort(-sims)[: max(top_k * 3, 10)]
            for idx in order:
                sim = float(sims[idx])
                if sim <= 0.2:
                    continue
                mid = self._ids[idx]
                if mid not in by_id:
                    continue
                prev, why = scores.get(mid, (0.0, set()))
                scores[mid] = (prev + sim, why | {"meaning"})

        hits = []
        for mid, (score, why) in scores.items():
            m = by_id[mid]
            if domain and m.domain != domain:
                continue
            hits.append(
                Hit(memory=m, score=score, why="both" if len(why) == 2 else next(iter(why)))
            )
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]
