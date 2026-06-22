"""환경변수 기반 전역 설정. `.env` 자동 로드."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # GCP 공통
    GCP_PROJECT_ID: str
    GCP_LOCATION: str = "us-central1"
    GCS_STAGING_BUCKET: str
    GCS_DATA_BUCKET: str

    # 공공데이터포털 e약은요 API 키 (URL-encoded 원본 그대로)
    DATA_API_KEY: str = ""

    # Vertex AI 모델
    VERTEX_MODEL_GEMINI: str = "gemini-2.5-flash"
    VERTEX_EMBED_MODEL: str = "text-multilingual-embedding-002"

    # Vertex AI Search
    SEARCH_DATA_STORE_ID: str = "med-rag-drugs"
    SEARCH_ENGINE_ID: str = "med-rag-engine"
    SEARCH_COLLECTION: str = "default_collection"

    # Vector Search
    VECTOR_INDEX_ID: str = ""
    VECTOR_INDEX_ENDPOINT_ID: str = ""
    VECTOR_DEPLOYED_INDEX_ID: str = "med_rag_v1"

    # FDA DUR API
    FDA_DUR_BASE_URL: str = "https://api.fda.gov/drug"

    # Agent Engine
    AGENT_ENGINE_RESOURCE_NAME: str = ""

    # UI
    UI_BACKEND_URL: str = Field(default="", description="Streamlit이 호출할 Agent Engine 엔드포인트")


settings = Settings()
