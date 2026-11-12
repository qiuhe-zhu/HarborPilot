import sys
import asyncio
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client.models import PointStruct

from app.config import BASE_DIR, QDRANT_COLLECTION
from app.rag import ensure_collection, get_qdrant
from app.llm import embed

DOCS_DIR = BASE_DIR / "data/docs"
CHUNK_SIZE = 500
OVERLAP = 80

def chunks(text: str):
    text = " ".join(text.split())
    out = []
    i = 0
    while i < len(text):
        out.append(text[i:i + CHUNK_SIZE])
        i += CHUNK_SIZE - OVERLAP
    return out

async def main():
    ensure_collection()
    client = get_qdrant()

    files = sorted(DOCS_DIR.rglob("*.md")) + sorted(DOCS_DIR.rglob("*.txt"))
    if not files:
        raise SystemExit(f"no docs found in {DOCS_DIR}")

    print(f"docs: {len(files)}")

    items = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(BASE_DIR))
        for idx, ch in enumerate(chunks(text)):
            if not ch.strip():
                continue
            items.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{path}:{idx}")),
                "source": rel,
                "chunk_id": idx,
                "text": ch,
            })

    BATCH = 32
    total = 0
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        vectors = await embed([x["text"] for x in batch])
        points = [
            PointStruct(
                id=x["id"],
                vector=v,
                payload={
                    "source": x["source"],
                    "chunk_id": x["chunk_id"],
                    "text": x["text"],
                },
            )
            for x, v in zip(batch, vectors)
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        total += len(points)
        print(f"upserted {total}/{len(items)}")

    print("done:", total)

if __name__ == "__main__":
    asyncio.run(main())
