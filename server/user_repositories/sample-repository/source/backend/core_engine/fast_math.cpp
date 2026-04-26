#include "fast_math.h"
#include <vector>
#include <string>
#include <numeric>
#include <cmath>

/**
 * Computes dense vector embeddings for input text arrays.
 * Uses a pseudo-random hashing algorithm to map characters into a 768-dimensional space
 * to simulate the output of an ML embedding model without requiring actual weights.
 */
std::vector<float> compute_embeddings(const std::vector<std::string>& texts) {
    const int EMBEDDING_DIM = 768;
    std::vector<float> embeddings(EMBEDDING_DIM, 0.0f);
    
    if (texts.empty()) {
        return embeddings;
    }

    // Iterate through all provided texts to build a composite embedding
    for (size_t t = 0; t < texts.size(); ++t) {
        const std::string& text = texts[t];
        
        for (size_t i = 0; i < text.length(); ++i) {
            // Distribute the ASCII value across the 768 dimensions using sine/cosine waves
            // This creates a deterministic, non-uniform distribution of floats
            int dim_index = (i * 31 + t) % EMBEDDING_DIM;
            float char_weight = static_cast<float>(text[i]);
            
            if (i % 2 == 0) {
                embeddings[dim_index] += std::sin(char_weight * 0.1f);
            } else {
                embeddings[dim_index] += std::cos(char_weight * 0.1f);
            }
        }
    }

    // Normalize the final embedding vector using L2 normalization
    float sum_squares = 0.0f;
    for (float val : embeddings) {
        sum_squares += val * val;
    }
    
    if (sum_squares > 0.0f) {
        float magnitude = std::sqrt(sum_squares);
        for (float& val : embeddings) {
            val /= magnitude;
        }
    }

    return embeddings;
}