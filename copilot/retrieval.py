"""Hybrid retrieval: BM25 (keyword) + TF-IDF cosine (dense-lite), fused with Reciprocal Rank Fusion,
then a light re-ranker. Runs offline with numpy + scikit-learn only.

To use real embeddings, swap `_dense_scores` for an embedding model + FAISS/Chroma index.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
STOP = set("a an the is are be by to of and or in on for with what how who when which does do can must "
           "need needs i we our my it its as at from after before than within this that".split())


def tokenize(text: str):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    heading: str
    text: str


def chunk_markdown(path: Path):
    """Split a markdown doc by '##' sections; each sentence-line becomes a chunk with its heading kept."""
    chunks, heading = [], ""
    title = path.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("## "):
            heading = line[3:].strip()
            continue
        n += 1
        chunks.append(Chunk(f"{path.stem}#{n}", title, heading, line))
    return chunks


class HybridRetriever:
    def __init__(self, chunks, k1: float = 1.4, b: float = 0.75):
        self.chunks = chunks
        self.texts = [f"{c.doc} {c.heading} {c.text}" for c in chunks]
        self.toks = [tokenize(t) for t in self.texts]
        self.k1, self.b = k1, b
        self.avgdl = sum(len(t) for t in self.toks) / max(len(self.toks), 1)
        df = Counter(w for t in self.toks for w in set(t))
        n = len(self.toks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.tf = [Counter(t) for t in self.toks]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=list(STOP))
        self.mat = self.vec.fit_transform(self.texts)

    @classmethod
    def from_dir(cls, docs_dir: Path):
        chunks = []
        for p in sorted(Path(docs_dir).glob("*.md")):
            chunks.extend(chunk_markdown(p))
        return cls(chunks)

    def _bm25(self, q):
        qt = tokenize(q)
        scores = np.zeros(len(self.toks))
        for i, tf in enumerate(self.tf):
            dl = len(self.toks[i])
            for w in qt:
                if w in tf:
                    f = tf[w]
                    scores[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return scores

    def _dense_scores(self, q):
        return (self.mat @ self.vec.transform([q]).T).toarray().ravel()

    def search(self, query: str, k: int = 3, rrf_k: int = 60):
        bm, de = self._bm25(query), self._dense_scores(query)
        fused = np.zeros(len(self.chunks))
        for scores in (bm, de):
            for rank, idx in enumerate(np.argsort(-scores)):
                fused[idx] += 1.0 / (rrf_k + rank + 1)
        cand = list(np.argsort(-fused)[: max(k * 3, 8)])
        # re-rank: fused score + bonus for query terms hitting the chunk text itself (not only title)
        qt = set(tokenize(query))
        def rerank(i):
            overlap = len(qt & set(tokenize(self.chunks[i].text))) / max(len(qt), 1)
            return fused[i] + 0.02 * overlap
        cand.sort(key=rerank, reverse=True)
        top = cand[:k]
        return [(self.chunks[i], float(de[i]), float(bm[i])) for i in top]
