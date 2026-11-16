"""Retrieval methods for Day 7, all over the same 148 Day 6 chunks (strategy D index).

1. DenseRetriever   - query embedding (multilingual-e5-small, 'query: ' prefix) -> Day 6 Chroma
                      collection (cosine HNSW), metadata filter applied inside Chroma.
2. BM25Retriever    - Okapi BM25 over syllable unigrams (+ optional syllable bigrams, because
                      Vietnamese words are mostly 2 syllables: 'động từ', 'tần suất').
3. HybridRetriever  - Reciprocal Rank Fusion of dense and BM25 candidate lists (rank based,
                      so no score normalisation is needed).
4. RerankRetriever  - any first stage -> top-N candidates re-scored by a multilingual cross-encoder.

Every result: id, text, source_id, source, page, metadata, score, rank, strategy (+ per-stage
ranks for fused/reranked results). Errors propagate; there is no mock fallback.
"""
import math
import re
import unicodedata
from collections import Counter
from typing import Any

from .common import CHROMA_PATH, DAY6_COLLECTION, EMBED_DIM, EMBED_MODEL


def matches(meta: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    return all(meta.get(k) == v for k, v in (filters or {}).items())


def _result(chunk: dict[str, Any], score: float, rank: int, strategy: str, **extra) -> dict[str, Any]:
    return {"id": chunk["id"], "text": chunk["text"], "source_id": chunk["source_id"], "source": chunk["source"],
            "page": chunk["page"], "metadata": chunk["metadata"], "score": round(float(score), 6),
            "rank": rank, "strategy": strategy, **extra}


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[dict[str, Any]]):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.by_id = {c["id"]: c for c in chunks}
        self.model = SentenceTransformer(EMBED_MODEL, device="cpu")
        dim = (getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension)()
        if dim != EMBED_DIM:
            raise RuntimeError(f"query embedding dim {dim} != index dim {EMBED_DIM}")
        self.col = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(DAY6_COLLECTION)
        if self.col.count() != len(chunks):
            raise RuntimeError(f"Chroma has {self.col.count()} chunks but export has {len(chunks)}; re-run export_day6")

    @staticmethod
    def _where(filters):
        if not filters:
            return None
        clauses = [{k: v} for k, v in filters.items()]
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        vec = self.model.encode(["query: " + query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
        res = self.col.query(query_embeddings=[vec], n_results=top_k, where=self._where(filters),
                             include=["distances"])
        return [_result(self.by_id[cid], 1 - dist, r + 1, self.name)
                for r, (cid, dist) in enumerate(zip(res["ids"][0], res["distances"][0]))]


def tokenize(text: str, bigrams: bool = True) -> list[str]:
    words = re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])] if bigrams else words


class BM25Retriever:
    def __init__(self, chunks: list[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75, bigrams: bool = True):
        self.chunks, self.k1, self.b, self.bigrams = chunks, k1, b, bigrams
        self.name = "bm25" if bigrams else "bm25_uni"
        self.tf = [Counter(tokenize(c["text"], bigrams)) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(term for t in self.tf for term in t)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q = Counter(tokenize(query, self.bigrams))
        scored = []
        for chunk, tf, ln in zip(self.chunks, self.tf, self.len):
            if not matches(chunk["metadata"], filters):
                continue
            s = 0.0
            for term in q:
                f = tf.get(term)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            if s > 0:
                scored.append((s, chunk))
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [_result(c, s, r + 1, self.name) for r, (s, c) in enumerate(scored[:top_k])]


class HybridRetriever:
    """RRF: score(d) = sum_i 1 / (k + rank_i(d)) over the dense and BM25 candidate lists."""

    def __init__(self, dense, lexical, *, candidates: int = 20, k: int = 60):
        self.dense, self.lexical, self.candidates, self.k = dense, lexical, candidates, k
        self.name = f"hybrid_rrf({dense.name}+{lexical.name})"

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        lists = {"dense": self.dense.search(query, filters=filters, top_k=self.candidates),
                 "bm25": self.lexical.search(query, filters=filters, top_k=self.candidates)}
        fused: dict[str, dict[str, Any]] = {}
        for stage, rows in lists.items():
            for r in rows:
                item = fused.setdefault(r["id"], {"row": r, "score": 0.0, "stage_ranks": {}})
                item["score"] += 1 / (self.k + r["rank"])
                item["stage_ranks"][stage] = r["rank"]
        ordered = sorted(fused.values(), key=lambda x: (-x["score"], x["row"]["id"]))[:top_k]
        return [_result(x["row"], x["score"], i + 1, self.name, stage_ranks=x["stage_ranks"])
                for i, x in enumerate(ordered)]


class RerankRetriever:
    """First stage -> `candidates` chunks -> cross-encoder score(query, chunk) -> top_k."""

    def __init__(self, first, *, model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", candidates: int = 20):
        from sentence_transformers import CrossEncoder

        self.first, self.candidates, self.model_name = first, candidates, model
        self.model = CrossEncoder(model, device="cpu", max_length=512)
        self.name = f"rerank({first.name})"

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        cands = self.first.search(query, filters=filters, top_k=self.candidates)
        if not cands:
            return []
        scores = self.model.predict([(query, c["text"]) for c in cands], show_progress_bar=False)
        ordered = sorted(zip(scores, cands), key=lambda x: (-float(x[0]), x[1]["id"]))[:top_k]
        return [_result(c, s, i + 1, self.name, first_stage_rank=c["rank"],
                        stage_ranks={**c.get("stage_ranks", {}), "first": c["rank"]})
                for i, (s, c) in enumerate(ordered)]


def build(method: str, chunks: list[dict[str, Any]], cache: dict | None = None):
    """Factory with a shared cache so the embedding/cross-encoder models load once per process."""
    cache = {} if cache is None else cache

    def get(name, factory):
        if name not in cache:
            cache[name] = factory()
        return cache[name]

    dense = lambda: get("dense", lambda: DenseRetriever(chunks))
    bm25 = lambda: get("bm25", lambda: BM25Retriever(chunks, bigrams=True))
    if method == "dense":
        return dense()
    if method == "bm25":
        return bm25()
    if method == "bm25_uni":
        return get("bm25_uni", lambda: BM25Retriever(chunks, bigrams=False))
    if method == "hybrid":
        return get("hybrid", lambda: HybridRetriever(dense(), bm25()))
    if method == "rerank_hybrid":
        return get("rerank_hybrid", lambda: RerankRetriever(get("hybrid", lambda: HybridRetriever(dense(), bm25()))))
    if method == "rerank_dense":
        return get("rerank_dense", lambda: RerankRetriever(dense()))
    raise ValueError(f"unknown retrieval method {method!r}")


METHODS = ("dense", "bm25_uni", "bm25", "hybrid", "rerank_dense", "rerank_hybrid")
