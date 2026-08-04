"""Vertex AI Search (sparse+semantic) + Vector Search (dense KNN) → RRF 합성.

원본의 OpenSearch BM25 + Chroma dense 하이브리드와 동일 컨셉. 두 검색기 결과를
**Reciprocal Rank Fusion** (TREC 2009, Cormack et al.)로 결합한다.

    RRF(d) = Σ_i  1 / (k + rank_i(d))    (k = 60, 표준값)

`drug_search_agent`가 호출하는 단일 도구로 노출하여, 모델이 두 검색기 결과를
이중으로 평가할 필요 없이 통합 결과만 보고 답변하게 한다.
"""

from __future__ import annotations

from tools.vector_search import search_drugs_vector
from tools.vertex_search import search_drugs

RRF_K = 60          # 표준 상수. 클수록 상위 결과의 영향이 줄어듦
SOURCE_TOP_K = 20   # 각 검색기에서 가져올 후보 수
RESULT_TOP_K = 10   # 최종 반환 수


def _rrf_combine(
    sparse_hits: list[dict],
    dense_hits: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """ID 기준으로 두 결과의 순위를 합쳐 RRF 점수 부여."""
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    docs: dict[str, dict] = {}

    for rank, hit in enumerate(sparse_hits, start=1):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        sources.setdefault(doc_id, set()).add("sparse")
        docs.setdefault(doc_id, hit)

    for rank, hit in enumerate(dense_hits, start=1):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        sources.setdefault(doc_id, set()).add("dense")
        # dense 결과로 메타데이터 보강 (snippet 등 sparse에 없을 수 있음)
        if doc_id not in docs:
            docs[doc_id] = hit

    fused = []
    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        doc = dict(docs[doc_id])
        doc["rrf_score"] = round(score, 6)
        doc["matched_by"] = sorted(sources[doc_id])
        fused.append(doc)
    return fused


def hybrid_search_drugs(query: str, top_k: int = RESULT_TOP_K) -> list[dict]:
    """Vertex AI Search + Vector Search를 RRF로 합쳐 최상위 결과 반환.

    Returns:
        [{
            "id", "item_name", "company", "categories",
            "snippet", "content",
            "rrf_score": float,
            "matched_by": ["sparse"|"dense"|both]
        }, ...]
    """
    sparse_hits = search_drugs(query, top_k=SOURCE_TOP_K)
    try:
        dense_hits = search_drugs_vector(query, top_k=SOURCE_TOP_K)
    except Exception as exc:  # noqa: BLE001
        # Vector endpoint 일시 장애 시 sparse만으로 graceful degrade
        print(f"vector search 실패, sparse only로 fallback: {exc!r}")
        dense_hits = []

    fused = _rrf_combine(sparse_hits, dense_hits)
    return fused[:top_k]
