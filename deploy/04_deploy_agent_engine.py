"""ADK 루트 에이전트를 Vertex AI Agent Engine에 배포.

배포 후 출력되는 resource name을 `.env`의 AGENT_ENGINE_RESOURCE_NAME에 채울 것.

Usage:
    python deploy/04_deploy_agent_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import vertexai  # noqa: E402
from vertexai import agent_engines  # noqa: E402

from agents.root_agent import root_agent  # noqa: E402
from config.settings import settings  # noqa: E402

REQUIREMENTS = [
    "google-adk>=0.5.0",
    "google-cloud-aiplatform[agent_engines]>=1.71.0",
    "google-cloud-discoveryengine>=0.13.0",
    "google-cloud-storage>=2.18.0",
    "pydantic>=2.8.0",
    "pydantic-settings>=2.5.0",
    "httpx>=0.27.0",
    "requests>=2.32.0",
]

EXTRA_PACKAGES = ["config", "agents", "tools"]


def main() -> None:
    vertexai.init(
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
        staging_bucket=settings.GCS_STAGING_BUCKET,
    )

    env_vars = {
        "GCP_PROJECT_ID": settings.GCP_PROJECT_ID,
        "GCP_LOCATION": settings.GCP_LOCATION,
        "GCS_STAGING_BUCKET": settings.GCS_STAGING_BUCKET,
        "GCS_DATA_BUCKET": settings.GCS_DATA_BUCKET,
        "VERTEX_MODEL_GEMINI": settings.VERTEX_MODEL_GEMINI,
        "VERTEX_EMBED_MODEL": settings.VERTEX_EMBED_MODEL,
        "SEARCH_DATA_STORE_ID": settings.SEARCH_DATA_STORE_ID,
        "SEARCH_ENGINE_ID": settings.SEARCH_ENGINE_ID,
        "SEARCH_COLLECTION": settings.SEARCH_COLLECTION,
        "VECTOR_INDEX_ENDPOINT_ID": settings.VECTOR_INDEX_ENDPOINT_ID,
        "VECTOR_DEPLOYED_INDEX_ID": settings.VECTOR_DEPLOYED_INDEX_ID,
        "FDA_DUR_BASE_URL": settings.FDA_DUR_BASE_URL,
    }

    existing = settings.AGENT_ENGINE_RESOURCE_NAME.strip()
    if existing:
        print(f"기존 Agent Engine update (resource={existing}) — 5~10분 소요")
        remote_app = agent_engines.update(
            resource_name=existing,
            agent_engine=root_agent,
            requirements=REQUIREMENTS,
            extra_packages=EXTRA_PACKAGES,
            env_vars=env_vars,
        )
    else:
        print("Agent Engine 신규 배포 시작 (5~15분 소요)")
        remote_app = agent_engines.create(
            root_agent,
            display_name="med-rag-supervisor",
            description="MedAgent RAG (LangGraph → ADK 재작성)",
            requirements=REQUIREMENTS,
            extra_packages=EXTRA_PACKAGES,
            env_vars=env_vars,
        )

    print()
    print("=" * 60)
    print("배포 완료. 다음 값을 .env에 채우세요:")
    print(f"AGENT_ENGINE_RESOURCE_NAME={remote_app.resource_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
