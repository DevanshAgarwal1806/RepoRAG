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

def embed_functions(
    functions: List[FunctionNode],
    output_path: str = "embeddings.json",
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Generates embeddings for all FunctionNodes and writes them to a JSON file.
    No caching — always recomputes everything.
    """
    print(f"  Embedding {len(functions)} functions...")

    model = get_model()
    texts = [build_embedding_text(fn) for fn in functions]

    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        all_embeddings.extend(vecs.tolist())

        done = min(i + batch_size, len(texts))
        print(f"  Embedded {done}/{len(functions)}")

    # Create output dictionary
    output: List[dict] = []
    for fn, emb in zip(functions, all_embeddings):
        fn.embedding = emb          # still store in object
        output.append({
            "id": fn.id,
            "embedding": emb
        })

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Done. Saved embeddings → {output_path}")