import json
from sentence_transformers import SentenceTransformer

def generate_and_save_embeddings(input_filepath: str, output_filepath: str):
    """
    Simulates the end of Phase 1: Reads parsed code, generates vector embeddings,
    and saves a finalized payload for the retrieval system.
    """
    print("Loading data and initializing embedding model...")
    
    # 1. Load the existing dummy codebase
    with open(input_filepath, 'r') as file:
        corpus = json.load(file)

    # 2. Load a fast, lightweight embedding model
    # all-MiniLM-L6-v2 is an industry standard for fast, local semantic search
    model = SentenceTransformer('jinaai/jina-embeddings-v2-base-code')

    # 3. Extract the text we want to embed
    texts_to_embed = [doc["text"] for doc in corpus]

    # 4. Generate the embeddings (This turns the text into arrays of numbers)
    print(f"Generating vectors for {len(corpus)} functions...")
    embeddings = model.encode(texts_to_embed).tolist()

    # 5. Inject the embeddings back into the corpus dictionary
    for i, doc in enumerate(corpus):
        doc["embedding"] = embeddings[i]

    # 6. Save the finalized corpus
    with open(output_filepath, 'w') as file:
        json.dump(corpus, file, indent=4)
        
    print(f"Success! Embedded corpus saved to {output_filepath}")

if __name__ == "__main__":
    generate_and_save_embeddings("dummy_data.json", "embedded_data.json")