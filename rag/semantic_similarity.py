"""
Embedding-based semantic similarity for PYQ intelligence.

ADVISORY SIGNAL ONLY. Embeddings may widen candidate admission and refine
confidence; they can never create an exact repeat, bypass structural vetoes
(entities / intent / constraints / requested output), or override source
grounding. When the model is unavailable every helper degrades to None/0.0
and callers fall back to deterministic lexical behaviour.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

_MODEL = None
_MODEL_LOCK = threading.Lock()
_LOAD_FAILED = False


def _get_model():
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None or _LOAD_FAILED:
            return _MODEL
        try:
            from sentence_transformers import SentenceTransformer

            from rag.config import EMBEDDING_MODEL_NAME

            _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as ex:
            print(f"[SEMANTIC_SIMILARITY] embedding model unavailable: {ex}")
            _LOAD_FAILED = True
            _MODEL = None
    return _MODEL


def embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    model = _get_model()
    if model is None or not texts:
        return None
    try:
        vecs = model.encode(
            [t if t and t.strip() else "empty question" for t in texts],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype=np.float32)
    except Exception as ex:
        print(f"[SEMANTIC_SIMILARITY] encode failed: {ex}")
        return None


def pair_cosine(matrix: Optional[np.ndarray], i: int, j: int) -> Optional[float]:
    if matrix is None:
        return None
    if i < 0 or j < 0 or i >= len(matrix) or j >= len(matrix):
        return None
    try:
        return float(np.dot(matrix[i], matrix[j]))
    except Exception:
        return None


def cosine_matrix_pairs(matrix: Optional[np.ndarray], pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}
    if matrix is None:
        return out
    for i, j in pairs:
        sim = pair_cosine(matrix, i, j)
        if sim is not None:
            out[(i, j)] = sim
    return out
