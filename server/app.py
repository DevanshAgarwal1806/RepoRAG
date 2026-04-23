from __future__ import annotations

import json
import re
import shutil
import sys
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from server.indexer_pipeline import run_indexer_pipeline
from retriever.bm25_basic import tokenize_code
from retriever.generator import generate_rag_answer
from retriever.graph_context import assemble_llm_context

USER_REPOS_DIR = APP_DIR / "user_repositories"
USER_REPOS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RepoRAG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repo_locks: dict[str, threading.Lock] = {}


class QueryRequest(BaseModel):
    query: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or f"repository-{int(datetime.now().timestamp())}"


def get_repo_lock(repo_id: str) -> threading.Lock:
    if repo_id not in repo_locks:
        repo_locks[repo_id] = threading.Lock()
    return repo_locks[repo_id]


def repo_dir(repo_id: str) -> Path:
    path = USER_REPOS_DIR / repo_id
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail="Repository not found.")
    return path


def metadata_path(path: Path) -> Path:
    return path / "metadata.json"


def source_dir(path: Path) -> Path:
    return path / "source"


def output_files(path: Path) -> dict[str, Path]:
    return {
        "functions": path / "extracted_functions.json",
        "graph": path / "dependency_graph.json",
        "embeddings": path / "embeddings.json",
        "imports": path / "import_map.json",
        "payload": path / "final_llm_payload.md",
    }


def read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_metadata(path: Path, values: dict[str, Any]) -> None:
    with open(metadata_path(path), "w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2)


def read_metadata(path: Path) -> dict[str, Any]:
    file_path = metadata_path(path)
    if not file_path.exists():
        return {
            "id": path.name,
            "name": path.name,
            "status": infer_status(path),
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }

    return read_json_file(file_path, {})


def infer_status(path: Path) -> str:
    files = output_files(path)
    if files["functions"].exists() and files["graph"].exists() and files["embeddings"].exists():
        return "indexed"
    if source_dir(path).exists():
        return "uploaded"
    return "error"


def summarize_repository(path: Path) -> dict[str, Any]:
    metadata = read_metadata(path)
    files = output_files(path)
    functions = read_json_file(files["functions"], [])

    return {
        "id": metadata.get("id", path.name),
        "name": metadata.get("name", path.name),
        "status": metadata.get("status", infer_status(path)),
        "updated_at": metadata.get("updated_at"),
        "created_at": metadata.get("created_at"),
        "error_message": metadata.get("error_message"),
        "function_count": len(functions),
        "embedding_ready": files["embeddings"].exists(),
    }


def list_repositories() -> list[dict[str, Any]]:
    repositories = [summarize_repository(path) for path in USER_REPOS_DIR.iterdir() if path.is_dir()]
    repositories.sort(key=lambda repo: repo.get("updated_at") or "", reverse=True)
    return repositories


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            resolved = destination / member.filename
            if not resolved.resolve().is_relative_to(destination.resolve()):
                raise HTTPException(status_code=400, detail="Invalid ZIP archive.")
        archive.extractall(destination)


def set_repo_status(path: Path, status: str, error_message: str | None = None) -> None:
    metadata = read_metadata(path)
    metadata["status"] = status
    metadata["error_message"] = error_message
    metadata["updated_at"] = now_iso()
    write_metadata(path, metadata)


def start_indexing(path: Path) -> None:
    def worker() -> None:
        lock = get_repo_lock(path.name)
        if not lock.acquire(blocking=False):
            return

        try:
            set_repo_status(path, "indexing")
            run_indexer_pipeline(str(source_dir(path)), str(path))
            set_repo_status(path, "indexed")
        except Exception as error:  # noqa: BLE001
            set_repo_status(path, "error", str(error))
        finally:
            lock.release()

    threading.Thread(target=worker, daemon=True).start()


def ensure_indexed_repository(path: Path) -> None:
    status = read_metadata(path).get("status", infer_status(path))
    if status != "indexed":
        raise HTTPException(status_code=409, detail="Repository is not ready for querying yet.")


def load_functions(path: Path) -> list[dict[str, Any]]:
    return read_json_file(output_files(path)["functions"], [])


def score_functions(path: Path, query: str, limit: int = 4) -> list[dict[str, Any]]:
    functions = load_functions(path)
    if not functions:
        return []

    tokenized_corpus = [tokenize_code(item.get("source_code", "")) for item in functions]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenize_code(query))
    ranked = sorted(zip(functions, scores), key=lambda item: item[1], reverse=True)
    return [item[0] for item in ranked[:limit]]


def build_fallback_answer(query: str, references: list[dict[str, Any]]) -> str:
    if not references:
        return (
            f"I could not find strong matches for '{query}' in the indexed repository. "
            "Try asking about a specific file, function, or subsystem."
        )

    lead = references[0]
    summary = [
        f"The strongest match for '{query}' is `{lead['name']}` in `{lead['file_path']}`.",
        "Relevant references are listed below so you can inspect the exact implementation quickly.",
    ]
    return " ".join(summary)


def query_repository(path: Path, query: str) -> dict[str, Any]:
    references = score_functions(path, query)
    reference_ids = [item["id"] for item in references]

    if reference_ids:
        payload, _ = assemble_llm_context(reference_ids, str(path), d=1)
        with open(output_files(path)["payload"], "w", encoding="utf-8") as handle:
            handle.write(f"### USER QUERY: {query}\n\n")
            handle.write(payload)

    answer = build_fallback_answer(query, references)
    generated_with_model = False

    try:
        answer = generate_rag_answer(str(path)) or answer
        generated_with_model = True
    except Exception:
        generated_with_model = False

    return {
        "answer": answer,
        "generated_with_model": generated_with_model,
        "references": [
            {
                "id": item["id"],
                "name": item["name"],
                "file_path": item["file_path"],
                "start_line": item["start_line"],
                "end_line": item["end_line"],
                "source_code": item["source_code"],
            }
            for item in references
        ],
    }


@app.get("/api/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/repositories")
def get_repositories() -> dict[str, Any]:
    return {"repositories": list_repositories()}


@app.post("/api/repositories/upload")
async def upload_repository(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or ""
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP uploads are supported.")

    repo_name = Path(filename).stem
    repo_id = slugify(repo_name)
    path = USER_REPOS_DIR / repo_id
    suffix = 1

    while path.exists():
        repo_id = f"{slugify(repo_name)}-{suffix}"
        path = USER_REPOS_DIR / repo_id
        suffix += 1

    path.mkdir(parents=True, exist_ok=True)
    src_dir = source_dir(path)
    src_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": repo_id,
        "name": repo_name,
        "status": "uploaded",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "error_message": None,
    }
    write_metadata(path, metadata)

    with NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await file.read())

    try:
        safe_extract_zip(temp_path, src_dir)
    except Exception:
        shutil.rmtree(path, ignore_errors=True)
        raise
    finally:
        temp_path.unlink(missing_ok=True)

    start_indexing(path)
    return {"repository": summarize_repository(path)}


@app.get("/api/repositories/{repo_id}")
def get_repository(repo_id: str) -> dict[str, Any]:
    path = repo_dir(repo_id)
    return summarize_repository(path)


@app.post("/api/repositories/{repo_id}/reindex")
def reindex_repository(repo_id: str) -> dict[str, Any]:
    path = repo_dir(repo_id)
    if not source_dir(path).exists():
        raise HTTPException(status_code=400, detail="Repository source files are missing.")

    start_indexing(path)
    return {"repository": summarize_repository(path)}


@app.post("/api/repositories/{repo_id}/query")
def ask_repository(repo_id: str, request: QueryRequest) -> dict[str, Any]:
    path = repo_dir(repo_id)
    ensure_indexed_repository(path)

    cleaned_query = request.query.strip()
    if not cleaned_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return query_repository(path, cleaned_query)


@app.on_event("startup")
def bootstrap_sample_repository() -> None:
    sample_path = USER_REPOS_DIR / "sample-repository"
    if sample_path.exists():
        return

    original_source = APP_DIR / "sample_repository"
    original_output = APP_DIR / "sample_repository_output"
    if not original_source.exists() or not original_output.exists():
        return

    sample_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(original_source, source_dir(sample_path))

    for filename in [
        "extracted_functions.json",
        "dependency_graph.json",
        "embeddings.json",
        "import_map.json",
        "final_llm_payload.md",
    ]:
        shutil.copy2(original_output / filename, sample_path / filename)

    write_metadata(
        sample_path,
        {
            "id": "sample-repository",
            "name": "Sample Repository",
            "status": "indexed",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "error_message": None,
        },
    )
