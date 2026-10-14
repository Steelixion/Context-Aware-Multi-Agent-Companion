"""On-device text embeddings (384-d). fastembed/ONNX if available, else hashed char-trigrams."""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

DIM = 384


class Embedder:
    def __init__(self, model_name: str):
        self.model = None
        self.name = "hash-fallback"
        try:
            from fastembed import TextEmbedding
            self.model = TextEmbedding(model_name=model_name)
            self.name = model_name
        except Exception as e:  # noqa: BLE001
            print(f"[embed] fastembed unavailable ({e}); using hashed-trigram fallback")

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        if self.model is not None:
            return [list(map(float, v)) for v in self.model.embed(list(texts))]
        return [self._hash(t) for t in texts]

    @staticmethod
    def _hash(text: str) -> list[float]:
        v = [0.0] * DIM
        t = f"  {text.lower()}  "
        for i in range(len(t) - 2):
            h = int(hashlib.md5(t[i:i + 3].encode()).hexdigest()[:8], 16)
            v[h % DIM] += 1.0 if (h >> 31) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]
