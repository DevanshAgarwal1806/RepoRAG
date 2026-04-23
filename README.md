# RepoRAG

## Run Frontend

```bash
cd client
npm install
npm run dev
```

The Vite dev server proxies `/api` requests to `http://127.0.0.1:8000`.

## Run Backend

```bash
pip install -r requirements.txt
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
