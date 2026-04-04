import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent.parent

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# 공공데이터포털
DATA_API_KEY = os.getenv("DATA_API_KEY")
DATA_API_BASE_URL = "http://apis.data.go.kr/1471000"

# ChromaDB
CHROMA_DB_PATH = str(PROJECT_ROOT / "chroma_db")
CHROMA_COLLECTION_DRUGS = "drugs"
CHROMA_COLLECTION_INTERACTIONS = "interactions"
CHROMA_COLLECTION_SAFETY = "safety"
