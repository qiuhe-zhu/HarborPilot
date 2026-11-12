import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

def _abs_path(p: str) -> str:
    pp = Path(p)
    return str(pp if pp.is_absolute() else BASE_DIR / pp)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "").rstrip("/")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

QDRANT_MODE = os.getenv("QDRANT_MODE", "local").lower()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
QDRANT_LOCAL_PATH = _abs_path(os.getenv("QDRANT_LOCAL_PATH", "data/qdrant_local"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "harbor_kb")
TOP_K = int(os.getenv("TOP_K", "5"))
