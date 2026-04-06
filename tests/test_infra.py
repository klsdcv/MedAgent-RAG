"""인프라 상태 검증 테스트.

Triton, ChromaDB, OpenSearch가 정상 동작하는지 확인.

Usage:
    pytest tests/test_infra.py -v
"""

import requests
import pytest

import chromadb
from src.config.settings import CHROMA_DB_PATH, CHROMA_COLLECTION_DRUGS, TRITON_URL


def test_triton_health():
    """Triton 서버가 살아있고 bge_m3 모델이 ready 상태인지."""
    resp = requests.get(f"{TRITON_URL}/v2/models/bge_m3/ready", timeout=5)
    assert resp.status_code == 200


def test_triton_inference():
    """Triton에서 임베딩이 실제로 나오는지."""
    from src.vectorstore.triton_embedder import TritonEmbedder

    embedder = TritonEmbedder()
    vec = embedder.embed_query("테스트 문장")
    assert len(vec) == 1024
    assert abs(sum(v**2 for v in vec) - 1.0) < 0.01  # L2 정규화 확인


def test_opensearch_health():
    """OpenSearch 클러스터가 정상인지."""
    resp = requests.get("http://localhost:9200/_cluster/health", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("green", "yellow")


def test_opensearch_drugs_index():
    """OpenSearch에 drugs 인덱스가 존재하고 문서가 있는지."""
    resp = requests.get("http://localhost:9200/drugs/_count", timeout=5)
    assert resp.status_code == 200
    count = resp.json()["count"]
    assert count > 0, "drugs 인덱스에 문서가 없음"


def test_chromadb_drugs_collection():
    """ChromaDB에 drugs 컬렉션이 존재하고 문서가 있는지."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(CHROMA_COLLECTION_DRUGS)
    assert collection.count() > 0, "drugs 컬렉션에 문서가 없음"


def test_hybrid_search_e2e():
    """하이브리드 검색 (Vector + BM25)이 결과를 반환하는지."""
    from src.agents.drug_search import hybrid_search

    results = hybrid_search("타이레놀")
    assert len(results) > 0
    assert "item_name" in results[0]["metadata"]
