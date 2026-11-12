import time
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from . import config, llm

_client = None

def get_qdrant():
    global _client
    if _client is None:
        if config.QDRANT_MODE == "local":
            _client = QdrantClient(path=config.QDRANT_LOCAL_PATH)
        else:
            _client = QdrantClient(url=config.QDRANT_URL)
    return _client

def ensure_collection():
    client = get_qdrant()
    if not client.collection_exists(config.QDRANT_COLLECTION):
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

async def retrieve(query: str, top_k: int | None = None):
    ensure_collection()
    client = get_qdrant()
    vec = (await llm.embed([query]))[0]

    # 新版本 qdrant-client 使用 query_points 方法
    response = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=vec,
        limit=top_k or config.TOP_K,
        with_payload=True,
    )

    # query_points 返回的对象里，命中结果在 .points 属性中
    hits = response.points

    return [
        {
            "id": str(h.id),
            "score": float(h.score),
            "source": h.payload.get("source"),
            "text": h.payload.get("text"),
        }
        for h in hits
    ]

def build_messages(question, contexts):
    context_text = "\n\n".join(
        f"[{i + 1}] source: {c['source']}\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    system = (
        "你是 HarborPilot，一个严谨的航运/海事问答助手。"
        "只能依据给定资料回答；资料不足就说不知道，不要编造。"
        "回答末尾用 [1][2] 标注引用。"
    )
    user = f"资料：\n{context_text}\n\n问题：{question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

async def ask(question: str):
    start = time.time()
    ensure_collection()
    contexts = await retrieve(question)
    messages = build_messages(question, contexts)
    answer = await llm.chat(messages)
    latency = int((time.time() - start) * 1000)
    return {
        "answer": answer,
        "sources": [
            {"source": c["source"], "score": round(c["score"], 4)}
            for c in contexts
        ],
        "latency_ms": latency,
        "trace_id": str(uuid.uuid4()),
    }