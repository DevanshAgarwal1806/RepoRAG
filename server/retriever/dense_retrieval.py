import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer('jinaai/jina-embeddings-v2-base-code')

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
    query_vector = _model.encode([user_query])[0].tolist()
    
    scores = []
    
    # 2. Compare the query vector to every function's vector
    for doc in embedded_corpus:
        score = cosine_similarity(query_vector, doc["embedding"])
        scores.append({"id": doc["id"], "score": score})
        
    # 3. Sort descending based on score
    scores.sort(key=lambda x: x["score"], reverse=True)
    
    # 4. Format the output with Explicit Ranks (1-based index)
    ranked_results = []
    for rank, item in enumerate(scores, start=1):
        ranked_results.append((item["id"], item["score"], rank))
        
    return ranked_results