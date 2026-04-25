import numpy as np
from typing import Any

MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
_model: Any | None = None


def get_model() -> Any:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    return _model

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculates the mathematical closeness of two vectors."""
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_dense_rankings(user_query: str, embedded_corpus: list[dict]) -> list[tuple[str, float, int]]:
    """
    Embeds the query and compares it to the corpus.
    Returns a list of tuples: (doc_id, score, rank)
    """
    # 1. Turn the user's query into a vector
    query_vector = get_model().encode(
        [user_query],
        normalize_embeddings=True
    )[0].tolist()
    
    scores = []
    
    # 2. Compare the query vector to every function's vector
    for doc in embedded_corpus:
        embedding = doc.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError(
                f"Embedding for {doc.get('id', '<unknown>')} is not a vector. "
                "Regenerate embeddings before dense retrieval."
            )
        score = cosine_similarity(query_vector, embedding)
        scores.append({"id": doc["id"], "score": score})
        
    # 3. Sort descending based on score
    scores.sort(key=lambda x: x["score"], reverse=True)
    
    # 4. Format the output with Explicit Ranks (1-based index)
    ranked_results = []
    for rank, item in enumerate(scores, start=1):
        ranked_results.append((item["id"], item["score"], rank))
        
    return ranked_results
