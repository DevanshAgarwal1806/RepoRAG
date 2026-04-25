import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from indexer.ast_parser import FunctionNode

MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
BATCH_SIZE = 1
CACHE_FILE = "embeddings_cache.json"

_model: Optional[SentenceTransformer] = None

def chunk_text(text: str, chunk_size_chars: int = 4000, overlap_chars: int = 400) -> list[str]:
    """Splits a long string into overlapping chunks."""
    if len(text) <= chunk_size_chars:
        return [text]
        
    chunks = []
    # Slide a window across the text
    for i in range(0, len(text), chunk_size_chars - overlap_chars):
        chunks.append(text[i : i + chunk_size_chars])
    return chunks

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Model: {MODEL_NAME}")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda") # For Nvidia GPUs
        else:
            device = torch.device("cpu")  # Fallback to standard CPU
        _model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        _model = _model.to(device)
        print(f"Device: {device}")
    return _model

def build_embedding_text(fn: FunctionNode) -> str:
    parts = []
    if fn.docstring:
        parts.append(f"Description: {fn.docstring}")
    parts.append(f"Function: {fn.name}")
    parts.append(fn.source_code)
    return "\n".join(parts)

def embed_functions(
    functions: List[FunctionNode],
    output_path: str = "embeddings.json"
) -> None:

    model = get_model()
    texts = [build_embedding_text(fn) for fn in functions]

    all_embeddings: List[List[float]] = []

    for fn in texts:
        # 1. Split the text into safe 4000-character (~1024 token) chunks
        chunks = chunk_text(fn)
        
        # 2. Embed all chunks for this specific function
        # We can pass the whole list of chunks; the model handles batching internally safely
        chunk_vecs = model.encode(
            chunks,
            batch_size=1,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        
        # 3. Combine them: Average the vectors across the column axis (axis=0)
        if len(chunks) > 1:
            combined_vec = np.mean(chunk_vecs, axis=0)
            
            # 4. CRITICAL: Re-normalize the combined vector so its length is 1.0
            norm = np.linalg.norm(combined_vec)
            if norm > 0:
                combined_vec = combined_vec / norm
        else:
            # If it was only 1 chunk, it's already normalized
            combined_vec = chunk_vecs[0]

        all_embeddings.extend(combined_vec.tolist())
        
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