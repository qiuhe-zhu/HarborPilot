from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from .rag import ask
from . import config

app = FastAPI(title="HarborPilot RAG API", version="0.1.0")

class AskReq(BaseModel):
    question: str
    top_k: int | None = None

@app.get("/health")
def health():
    return {"ok": True, "service": "harborpilot", "version": "0.1.0"}

@app.post("/ask")
async def ask_api(req: AskReq):
    try:
        result = await ask(req.question)
        return result
    except httpx.HTTPError as e:
        raise HTTPException(502, f"上游 LLM/Embedding 接口报错: {e}")
    except Exception as e:
        raise HTTPException(500, f"服务器内部错误: {e}")
