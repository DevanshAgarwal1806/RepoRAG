# embedder.py
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from ast_parser import FunctionNode

# ── Model ─────────────────────────────────────────────────────────────────────
# jina-embeddings-v2-base-code supports 8192 token context — long functions
# won't get silently truncated the way they would with most other models.
MODEL_NAME  = "jinaai/jina-embeddings-v2-base-code"
BATCH_SIZE  = 32        # safe default for CPU; increase to 64-128 on GPU
CACHE_FILE  = "embeddings_cache.json"

_model: Optional[SentenceTransformer] = None

def get_model() -> SentenceTransformer:
    """Lazy-load the model once and reuse it."""
    global _model
    if _model is None:
        print(f"  Loading model: {MODEL_NAME} ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        _model = _model.to(device)
        print(f"  Model loaded on {device.upper()}.")
    return _model

# ── Text construction ──────────────────────────────────────────────────────────
def build_embedding_text(fn: FunctionNode) -> str:
    """
    Combines docstring + name + source into a single string.
    Docstring is placed first so the model sees semantic intent before raw code.
    """
    parts = []
    if fn.docstring:
        parts.append(f"Description: {fn.docstring}")
    parts.append(f"Function: {fn.name}")
    parts.append(fn.source_code)
    return "\n".join(parts)

# ── Cache ──────────────────────────────────────────────────────────────────────
def load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache: dict, cache_path: str):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

# ── Core embedding ─────────────────────────────────────────────────────────────
def embed_functions(
    functions: List[FunctionNode],
    cache_path: str = CACHE_FILE,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Generates embeddings for all FunctionNodes and mutates fn.embedding in place.
    Skips functions whose embedding is already cached (keyed by fn.id).
    """
    cache = load_cache(cache_path)

    # Separate already-cached from needing embedding
    to_embed: List[FunctionNode] = []
    for fn in functions:
        if fn.id in cache:
            fn.embedding = cache[fn.id]
        else:
            to_embed.append(fn)

    if not to_embed:
        print(f"  All {len(functions)} embeddings loaded from cache.")
        return

    print(f"  {len(cache)} cached | {len(to_embed)} to embed ...")

    model  = get_model()
    texts  = [build_embedding_text(fn) for fn in to_embed]
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs  = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,   # unit vectors → cosine sim = dot product
            convert_to_numpy=True,
        )
        all_embeddings.extend(vecs.tolist())
        done = min(i + batch_size, len(texts))
        print(f"  Embedded {done}/{len(to_embed)}")

    # Write back to nodes and update cache
    for fn, emb in zip(to_embed, all_embeddings):
        fn.embedding = emb
        cache[fn.id] = emb

    save_cache(cache, cache_path)
    print(f"  Done. Cache updated → {cache_path}")

# ── Similarity utils ───────────────────────────────────────────────────────────
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

def find_similar(
    query_fn: FunctionNode,
    all_fns: List[FunctionNode],
    top_k: int = 5,
) -> List[tuple[FunctionNode, float]]:
    """Returns top_k most semantically similar functions to query_fn."""
    if query_fn.embedding is None:
        raise ValueError(f"query_fn '{query_fn.name}' has no embedding.")

    scored = [
        (fn, cosine_similarity(query_fn.embedding, fn.embedding))
        for fn in all_fns
        if fn.id != query_fn.id and fn.embedding is not None
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

def find_similar_to_query(
    query_text: str,
    all_fns: List[FunctionNode],
    top_k: int = 5,
) -> List[tuple[FunctionNode, float]]:
    """Embeds a free-text query and finds the most similar functions."""
    model = get_model()
    vec   = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].tolist()

    scored = [
        (fn, cosine_similarity(vec, fn.embedding))
        for fn in all_fns
        if fn.embedding is not None
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]