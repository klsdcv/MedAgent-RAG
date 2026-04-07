"""OpenSearch 키워드 검색 클라이언트."""

from opensearchpy import OpenSearch

OPENSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "drugs"


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        use_ssl=False,
        verify_certs=False,
    )


def create_index(client: OpenSearch | None = None):
    """drugs 인덱스 생성 (한국어 nori 분석기 포함)."""
    client = client or get_client()

    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)

    body = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "korean": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "filter": ["lowercase"],
                    }
                }
            },
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "item_name": {
                    "type": "text",
                    "analyzer": "korean",
                    "fields": {
                        "keyword": {"type": "keyword"},
                        "ngram": {
                            "type": "text",
                            "analyzer": "standard",
                        },
                    },
                },
                "company": {"type": "keyword"},
                "document": {
                    "type": "text",
                    "analyzer": "korean",
                },
                "item_seq": {"type": "keyword"},
                "update_date": {"type": "keyword"},
            }
        },
    }

    client.indices.create(index=INDEX_NAME, body=body)
    print(f"인덱스 '{INDEX_NAME}' 생성 완료")


def search(query: str, n_results: int = 10, index: str = INDEX_NAME, client: OpenSearch | None = None) -> list[dict]:
    """BM25 키워드 검색."""
    client = client or get_client()

    body = {
        "size": n_results,
        "query": {
            "multi_match": {
                "query": query,
                "fields": [
                    "item_name^3",
                    "item_name.ngram^2",
                    "document",
                ],
                "type": "best_fields",
            }
        },
    }

    resp = client.search(index=index, body=body)

    results = []
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        results.append({
            "id": src.get("item_seq", hit["_id"]),
            "document": src.get("document", ""),
            "metadata": {k: v for k, v in src.items() if k != "document"},
            "bm25_score": hit["_score"],
        })

    return results


SAFETY_INDEX = "safety"


def create_safety_index(client: OpenSearch | None = None):
    """safety 인덱스 생성 (임부금기, 연령대금기)."""
    client = client or get_client()

    if client.indices.exists(index=SAFETY_INDEX):
        client.indices.delete(index=SAFETY_INDEX)

    body = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "korean": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "filter": ["lowercase"],
                    }
                }
            },
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "item_name": {
                    "type": "text",
                    "analyzer": "korean",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "ingr_name": {
                    "type": "text",
                    "analyzer": "korean",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "type": {"type": "keyword"},
                "item_seq": {"type": "keyword"},
                "ingr_code": {"type": "keyword"},
                "company": {"type": "keyword"},
                "prohbt_content": {
                    "type": "text",
                    "analyzer": "korean",
                },
                "document": {
                    "type": "text",
                    "analyzer": "korean",
                },
            }
        },
    }

    client.indices.create(index=SAFETY_INDEX, body=body)
    print(f"인덱스 '{SAFETY_INDEX}' 생성 완료")


def search_safety(query: str, n_results: int = 5, client: OpenSearch | None = None) -> list[dict]:
    """safety 인덱스 BM25 검색."""
    return search(query, n_results=n_results, index=SAFETY_INDEX, client=client)
