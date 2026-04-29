import json
from sentence_transformers import SentenceTransformer

def generate_and_save_embeddings(input_filepath: str, output_filepath: str):
    print("Loading data and initializing embedding model...")
    
    with open(input_filepath, 'r') as file:
        corpus = json.load(file)

    model = SentenceTransformer('jinaai/jina-embeddings-v2-base-code')

    texts_to_embed = [doc["text"] for doc in corpus]

    print(f"Generating vectors for {len(corpus)} functions...")
    embeddings = model.encode(texts_to_embed).tolist()

    for i, doc in enumerate(corpus):
        doc["embedding"] = embeddings[i]

    with open(output_filepath, 'w') as file:
        json.dump(corpus, file, indent=4)
        
    print(f"Success! Embedded corpus saved to {output_filepath}")

if __name__ == "__main__":
    generate_and_save_embeddings("dummy_data.json", "embedded_data.json")