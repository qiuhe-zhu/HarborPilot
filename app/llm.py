import httpx
from . import config

def _require(name: str, value: str):
    if not value:
        raise RuntimeError(f"{name} 未配置，请检查 .env")

async def chat(messages, temperature=0.2):
    _require("LLM_BASE_URL", config.LLM_BASE_URL)
    _require("LLM_API_KEY", config.LLM_API_KEY)

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{config.LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            json={
                "model": config.LLM_MODEL,
                "messages": messages,
                "temperature": temperature,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def embed(texts):
    _require("EMBEDDING_BASE_URL", config.EMBEDDING_BASE_URL)
    _require("EMBEDDING_API_KEY", config.EMBEDDING_API_KEY)

    if isinstance(texts, str):
        texts = [texts]

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{config.EMBEDDING_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {config.EMBEDDING_API_KEY}"},
            json={"model": config.EMBEDDING_MODEL, "input": texts},
        )
        r.raise_for_status()
        data = r.json()["data"]
        return [d["embedding"] for d in data]
