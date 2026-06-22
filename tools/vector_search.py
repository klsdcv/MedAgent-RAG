"""Vertex Vector Search dense KNN 검색 툴.

`text-multilingual-embedding-002`로 질의를 임베딩 → Vector Search 인덱스에서
KNN 결과 반환. 결과는 ID만 돌아오므로, GCS의 중간 JSONL에서 메타데이터를
lazy load하여 약품명·본문을 함께 반환한다.
"""

from __future__ import annotations

import json
from functools import lru_cache

import vertexai
from google.cloud import aiplatform, storage
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from config.settings import settings

PROCESSED_GCS_KEY = "drugs/processed/drugs.jsonl"
EMBED_DIM = 768
MAX_QUERY_CHARS = 2000


@lru_cache(maxsize=1)
def _init():
    vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
    aiplatform.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
    return True


@lru_cache(maxsize=1)
def _embed_model() -> TextEmbeddingModel:
    _init()
    return TextEmbeddingModel.from_pretrained(settings.VERTEX_EMBED_MODEL)


@lru_cache(maxsize=1)
def _endpoint() -> aiplatform.MatchingEngineIndexEndpoint:
    _init()
    if not settings.VECTOR_INDEX_ENDPOINT_ID:
        raise RuntimeError("VECTOR_INDEX_ENDPOINT_ID 환경변수가 비어있습니다.")
    return aiplatform.MatchingEngineIndexEndpoint(settings.VECTOR_INDEX_ENDPOINT_ID)


@lru_cache(maxsize=1)
def _metadata_cache() -> dict[str, dict]:
    """id → {item_name, company, content, ...} 매핑."""
    client = storage.Client(project=settings.GCP_PROJECT_ID)
    bucket_name = settings.GCS_DATA_BUCKET.removeprefix("gs://").rstrip("/")
    blob = client.bucket(bucket_name).blob(PROCESSED_GCS_KEY)
    raw = blob.download_as_text(encoding="utf-8")

    cache: dict[str, dict] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cache[rec["id"]] = {
            "item_name": rec["structData"].get("item_name", ""),
            "company": rec["structData"].get("company", ""),
            "categories": rec["structData"].get("categories", []),
            "content": rec["content"],
        }
    return cache


def search_drugs_vector(query: str, top_k: int = 10) -> list[dict]:
    """질의 → 임베딩 → KNN 검색 → 메타데이터 결합 반환.

    Returns:
        [{"id", "item_name", "company", "categories", "distance", "snippet", "content"}, ...]
    """
    inputs = [
        TextEmbeddingInput(text=query[:MAX_QUERY_CHARS], task_type="RETRIEVAL_QUERY")
    ]
    vec = _embed_model().get_embeddings(inputs, output_dimensionality=EMBED_DIM)[0].values

    response = _endpoint().find_neighbors(
        deployed_index_id=settings.VECTOR_DEPLOYED_INDEX_ID,
        queries=[vec],
        num_neighbors=top_k,
    )

    cache = _metadata_cache()
    results: list[dict] = []
    for neighbor in response[0]:
        meta = cache.get(neighbor.id, {})
        content = meta.get("content", "")
        results.append(
            {
                "id": neighbor.id,
                "item_name": meta.get("item_name", ""),
                "company": meta.get("company", ""),
                "categories": meta.get("categories", []),
                "distance": float(neighbor.distance),
                "snippet": content[:300] + ("…" if len(content) > 300 else ""),
                "content": content,
            }
        )
    return results
