import json
import numpy as np
from pathlib import Path
import ollama
from sentence_transformers import SentenceTransformer

SERVER_DIR   = Path(__file__).resolve().parent.parent
EVALUATION_DIR = SERVER_DIR / "evaluation"
GROUND_TRUTH_DIR = EVALUATION_DIR / "0-ground_truth_construction"

SINGLEHOP_FILE = GROUND_TRUTH_DIR / "singlehop_ground_truth.json"
MULTIHOP_FILE  = GROUND_TRUTH_DIR / "multihop_ground_truth.json"

CLASSIFIER_MODEL = "gemma3:4b"
ENCODER_MODEL = "all-MiniLM-L6-v2"
NUM_SHOTS = 8

SYSTEM_PROMPT = """You are a query classifier for a code search system.
Classify the query as 'single-hop' or 'multi-hop':

single-hop: The query asks about ONE function/class in isolation.
            Answerable from that function's code.

multi-hop:  The query requires understanding RELATIONSHIPS between components.
            Needs to know callers, callees, dependencies, or inheritance chains.
            Examples: what calls X, how does A interact with B,
                      what depends on Y, trace the flow from Z,
                      where is this function used.

You MUST respond only with valid JSON in exactly this format:
{"label": "single-hop", "reasoning": "..."} 
or 
{"label": "multi-hop", "reasoning": "..."}"""

def load_examples() -> list[dict]:
    with open(SINGLEHOP_FILE, "r", encoding="utf-8") as f:
        single = json.load(f)

    with open(MULTIHOP_FILE, "r", encoding="utf-8") as f:
        multi = json.load(f)

    examples = []
    for item in single:
        examples.append({"query": item["query"], "label": "single-hop"})
    for item in multi:
        examples.append({"query": item["query"], "label": "multi-hop"})

    print(f"Loaded {len(single)} single-hop and {len(multi)} multi-hop examples")
    return examples

def build_example_store(examples: list[dict]) -> dict:
    print(f"Encoding {len(examples)} examples with '{ENCODER_MODEL}'...")
    encoder = SentenceTransformer(ENCODER_MODEL)
    queries = [e["query"] for e in examples]
    embeddings = encoder.encode(queries, normalize_embeddings=True, show_progress_bar=True)
    print("Example store ready.\n")
    return {
        "examples": examples,
        "embeddings": embeddings,
        "encoder": encoder,
    }

def get_similar_examples(query: str, store: dict, k: int = NUM_SHOTS) -> list[dict]:
    q_emb = store["encoder"].encode([query], normalize_embeddings=True)
    scores = (store["embeddings"] @ q_emb.T).squeeze()
    top_k = np.argsort(scores)[::-1]

    per_class = k // 2
    single, multi = [], []
    for idx in top_k:
        example = store["examples"][idx]
        if example["label"] == "single-hop" and len(single) < per_class:
            single.append(example)
        elif example["label"] == "multi-hop" and len(multi) < per_class:
            multi.append(example)
        if len(single) == per_class and len(multi) == per_class:
            break

    return single + multi

def build_few_shot_prompt(query: str, examples: list[dict]) -> str:
    shots = "\n\n".join(
        f"Query: {e['query']}\nLabel: {e['label']}"
        for e in examples
    )
    return f"{shots}\n\nQuery: {query}\nLabel:"

def classify_query(query: str, store: dict) -> dict:
    examples = get_similar_examples(query, store)
    user_prompt = build_few_shot_prompt(query, examples)

    response = ollama.chat(
        model=CLASSIFIER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        format="json",
        options={"temperature": 0.0},
    )

    result = json.loads(response.message.content)
    label = result.get("label", "").strip().lower()

    if label not in ("single-hop", "multi-hop"):
        raise ValueError(f"Unexpected label from model: '{label}'")

    return {
        "label": label,
        "reasoning": result.get("reasoning", ""),
    }

def evaluate(store: dict) -> None:
    from sklearn.metrics import classification_report

    examples = store["examples"]
    y_true, y_pred = [], []

    print("Running leave-one-out evaluation...\n")

    for i, item in enumerate(examples):
        # Temporarily exclude this item from the store
        loo_store = {
            "examples": examples[:i] + examples[i+1:],
            "embeddings": np.concatenate([
                store["embeddings"][:i], store["embeddings"][i+1:]
            ], axis=0),
            "encoder": store["encoder"],
        }
        try:
            result = classify_query(item["query"], loo_store)
            y_pred.append(result["label"])
        except Exception as e:
            print(f"  ERROR on item {i}: {e}")
            y_pred.append("single-hop")  # default fallback

        y_true.append(item["label"])

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(examples)} done...")

    print("Evaluation Results:")
    print(classification_report(y_true, y_pred, target_names=["single-hop", "multi-hop"]))

if __name__ == "__main__":
    examples = load_examples()
    store = build_example_store(examples)

    test_queries = [
        "What parameters does DummyTool.__init__ take?",
        "What functions call the retry handler and how does it propagate errors?",
    ]
    
    for q in test_queries:
        result = classify_query(q, store)
        print(f"Query    : {q}")
        print(f"Label    : {result['label']}")
        print(f"Reasoning: {result['reasoning']}\n")

    evaluate(store)