from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import run


class PromptRequest(BaseModel):
    user_prompt: str = Field(..., min_length=1, description="Prompt sent by the frontend")


class PromptResponse(BaseModel):
    current_dag: dict
    final_output: str | None


app = FastAPI(title="SynapseAI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=PromptResponse)
async def run_prompt(payload: PromptRequest) -> PromptResponse:
    try:
        result = await run_in_threadpool(run, payload.user_prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PromptResponse(
        current_dag=result.get("current_dag", {}),
        final_output=result.get("final_output"),
    )