"""Vertex AI Search (Discovery Engine) 키워드+시맨틱 검색 툴.

원본의 OpenSearch sparse 검색을 대체. 데이터스토어 `med-rag-drugs`는
data_schema="custom"으로 import되었고 각 문서는 `item_name`, `company`,
`update_date`, `categories`, `content` 필드를 가진다.
"""

from __future__ import annotations

from functools import lru_cache

from google.cloud import discoveryengine_v1 as de

from config.settings import settings

_LOCATION = "global"


@lru_cache(maxsize=1)
def _client() -> de.SearchServiceClient:
    return de.SearchServiceClient()


def _serving_config() -> str:
    return (
        f"projects/{settings.GCP_PROJECT_ID}"
        f"/locations/{_LOCATION}"
        f"/collections/{settings.SEARCH_COLLECTION}"
        f"/dataStores/{settings.SEARCH_DATA_STORE_ID}"
        f"/servingConfigs/default_search"
    )


def search_drugs(query: str, top_k: int = 10) -> list[dict]:
    """약품 데이터스토어를 검색하여 상위 결과 반환.

    Returns:
        [{"id", "item_name", "company", "categories", "snippet"}, ...]
        - snippet은 본문(content) 앞부분 (실제 snippet 추출은 본문 자체에서)
    """
    request = de.SearchRequest(
        serving_config=_serving_config(),
        query=query,
        page_size=top_k,
    )
    response = _client().search(request)

    results: list[dict] = []
    for r in response.results:
        doc = r.document
        sd = dict(doc.struct_data) if doc.struct_data else {}
        content = sd.get("content", "") or ""
        snippet = content[:300] + ("…" if len(content) > 300 else "")
        results.append(
            {
                "id": doc.id,
                "item_name": sd.get("item_name", ""),
                "company": sd.get("company", ""),
                "categories": list(sd.get("categories", []) or []),
                "snippet": snippet,
                "content": content,
            }
        )
    return results
