import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent.parent

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o"

# 임베딩 (BGE-M3 via Triton)
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
TRITON_URL = os.getenv("TRITON_URL", "http://localhost:8000")
TRITON_MODEL_NAME = "bge_m3"

# 공공데이터포털
DATA_API_KEY = os.getenv("DATA_API_KEY")
DATA_API_BASE_URL = "http://apis.data.go.kr/1471000"

# LangSmith
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "MedAgent-RAG")

# ChromaDB
CHROMA_DB_PATH = str(PROJECT_ROOT / "chroma_db")
CHROMA_COLLECTION_DRUGS = "drugs"
CHROMA_COLLECTION_INTERACTIONS = "interactions"
CHROMA_COLLECTION_SAFETY = "safety"
