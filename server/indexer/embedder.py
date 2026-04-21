# embedder.py
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from indexer.ast_parser import FunctionNode

MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
BATCH_SIZE = 32
CACHE_FILE = "embeddings_cache.json"

_model: Optional[SentenceTransformer] = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Model: {MODEL_NAME}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        _model = _model.to(device)
        print(f"Device: {device.upper()}")
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
        
    # Create output dictionary
    output: List[dict] = []
    for fn, emb in zip(functions, all_embeddings):
        fn.embedding = emb
        output.append({
            "id": fn.id,
            "embedding": emb
        })

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved embeddings to {output_path}")