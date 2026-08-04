"""Vector Search 인덱스 생성 → 엔드포인트 생성 → 인덱스 deploy.

이 스크립트는 long-running이며 약 30-60분 소요된다:
- 인덱스 빌드: 10-25분
- 엔드포인트 생성: 1-3분
- 인덱스 deploy: 15-30분

성공 시 마지막에 .env에 채워야 할 값을 출력한다.

Usage:
    python deploy/03_build_vector_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 패키지 root을 PYTHONPATH에 추가 (deploy/는 패키지가 아님)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from google.cloud import aiplatform  # noqa: E402

from config.settings import settings  # noqa: E402

CONTENTS_DELTA_URI = f"{settings.GCS_DATA_BUCKET.rstrip('/')}/drugs/vector_clean"
EMBED_DIM = 768
INDEX_DISPLAY_NAME = "med-rag-drugs-index"
ENDPOINT_DISPLAY_NAME = "med-rag-drugs-endpoint"


def init() -> None:
    aiplatform.init(
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
        staging_bucket=settings.GCS_STAGING_BUCKET,
    )


def create_or_get_index() -> aiplatform.MatchingEngineIndex:
    existing = aiplatform.MatchingEngineIndex.list(
        filter=f'display_name="{INDEX_DISPLAY_NAME}"'
    )
    if existing:
        idx = existing[0]
        print(f"인덱스 재사용: {idx.resource_name}")
        return idx

    print(f"인덱스 생성 시작 (contents={CONTENTS_DELTA_URI}) — 10~25분 소요")
    idx = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DISPLAY_NAME,
        description="MedAgent drugs (e약은요) vector index",
        contents_delta_uri=CONTENTS_DELTA_URI,
        dimensions=EMBED_DIM,
        approximate_neighbors_count=150,
        distance_measure_type="COSINE_DISTANCE",
        leaf_node_embedding_count=1000,
        leaf_nodes_to_search_percent=10,
        index_update_method="BATCH_UPDATE",
        shard_size="SHARD_SIZE_SMALL",
    )
    print(f"인덱스 생성 완료: {idx.resource_name}")
    return idx


def create_or_get_endpoint() -> aiplatform.MatchingEngineIndexEndpoint:
    existing = aiplatform.MatchingEngineIndexEndpoint.list(
        filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"'
    )
    if existing:
        ep = existing[0]
        print(f"엔드포인트 재사용: {ep.resource_name}")
        return ep

    print("엔드포인트 생성 시작 — public, 1~3분")
    ep = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        public_endpoint_enabled=True,
        description="MedAgent drugs vector search endpoint",
    )
    print(f"엔드포인트 생성 완료: {ep.resource_name}")
    return ep


def deploy_if_needed(
    endpoint: aiplatform.MatchingEngineIndexEndpoint,
    index: aiplatform.MatchingEngineIndex,
) -> None:
    deployed = [d for d in endpoint.deployed_indexes if d.id == settings.VECTOR_DEPLOYED_INDEX_ID]
    if deployed:
        print(f"이미 deploy됨: {settings.VECTOR_DEPLOYED_INDEX_ID}")
        return

    print(f"인덱스 deploy 시작 ({settings.VECTOR_DEPLOYED_INDEX_ID}) — 15~30분 소요")
    endpoint.deploy_index(
        index=index,
        deployed_index_id=settings.VECTOR_DEPLOYED_INDEX_ID,
        display_name=settings.VECTOR_DEPLOYED_INDEX_ID,
        machine_type="e2-standard-2",
        min_replica_count=1,
        max_replica_count=1,
    )
    print("deploy 완료")


def main() -> None:
    init()
    index = create_or_get_index()
    endpoint = create_or_get_endpoint()
    deploy_if_needed(endpoint, index)

    print()
    print("=" * 60)
    print(".env에 다음 값을 채우세요:")
    print(f"VECTOR_INDEX_ID={index.name}")
    print(f"VECTOR_INDEX_ENDPOINT_ID={endpoint.name}")
    print(f"VECTOR_DEPLOYED_INDEX_ID={settings.VECTOR_DEPLOYED_INDEX_ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()
