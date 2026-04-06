def calculate_bm25(tf, idf):
    """Calculates the BM25 score for a given term."""
    # Simplified for testing
    return tf * idf

def normalize_score(score):
    # Notice this function intentionally lacks a docstring
    return score / 100.0