# RepoRAG

## Run Frontend
Ensure that npm is installed and set up.
```bash
cd client
npm install
npm run dev
```

The Vite dev server proxies `/api` requests to `http://127.0.0.1:8000`.

## Run Backend
Ensure that a local python environment is setup to execute the commands below. Also ensure that `ollama` is installed and running.
```bash
pip install -r requirements.txt
ollama pull gemma4:3b
uvicorn server.app:app --reload
```

If you want the API on a specific host/port:

```bash
uvicorn server.app:app --reload --host 127.0.0.1 --port 8000
```

## CLI Pipeline

```bash
python server/run_pipeline.py --repo <repo_path> --query <query> --rerun
```
To prevent reindexing each time the query is run, remove the `--rerun` flag.
