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

The system uses GROQ API key. Create a `.env` file in the server folder and format it as `GROQ_API_KEY = <API-KEY>`

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
Ensure that the repository to query is in the same folder as the server folder.
```bash
python server/run_pipeline.py --repo <repo_path> --query <query> --rerun
```
To prevent reindexing each time the query is run, remove the `--rerun` flag.

## Note on Evaluation
Gemini API key has been used for evaluation. If you want to execute the evaluation scripts, ensure that the `.env` file contains `GEMINI_API_KEY = <API-KEY>`

To run generator evaluation, the following models are required.
```bash
ollama pull phi4-mini:3.8b 
ollama pull qwen2.5-coder:3b
```