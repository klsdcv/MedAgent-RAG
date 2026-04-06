"""Drug Search Agent - 하이브리드 검색 (ChromaDB Vector + OpenSearch BM25 + RRF + Rerank)."""

import chromadb

from src.config.settings import CHROMA_DB_PATH, CHROMA_COLLECTION_DRUGS
from src.graph.state import MedAgentState
from src.vectorstore.triton_embedder import TritonEmbedder
from src.vectorstore.reranker import Reranker
from src.vectorstore import opensearch_client


embedder = TritonEmbedder()
reranker = Reranker()


def search_vector(query: str, n_results: int = 10) -> list[dict]:
    """ChromaDB 벡터 검색."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(CHROMA_COLLECTION_DRUGS)

    query_embedding = embedder.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    items = []
    for doc_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        items.append({
            "id": doc_id,
            "document": doc,
            "metadata": meta,
            "vector_score": 1 - dist,
        })

    return items


def search_keyword(query: str, n_results: int = 10) -> list[dict]:
    """OpenSearch BM25 키워드 검색."""
    return opensearch_client.search(query, n_results=n_results)


def hybrid_search(query: str, n_results: int = 5, vector_weight: float = 0.6) -> list[dict]:
    """벡터 + OpenSearch BM25 하이브리드 검색 (RRF 기반 re-ranking).

    Reciprocal Rank Fusion으로 두 검색 결과를 합산.
    """
    vector_results = search_vector(query, n_results=n_results * 2)
    bm25_results = search_keyword(query, n_results=n_results * 2)

    # RRF 스코어 계산
    k = 60  # RRF 상수
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, item in enumerate(vector_results):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (k + rank + 1)
        doc_map[doc_id] = item

    bm25_weight = 1 - vector_weight
    for rank, item in enumerate(bm25_results):
        doc_id = item["id"]
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (k + rank + 1)
        if doc_id not in doc_map:
            doc_map[doc_id] = item

    # 스코어 기준 정렬
    ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:n_results]

    results = []
    for doc_id in ranked_ids:
        item = doc_map[doc_id]
        item["hybrid_score"] = scores[doc_id]
        results.append(item)

    return results


def drug_search_node(state: MedAgentState) -> dict:
    """Drug Search Agent 노드 함수."""
    query = state["query"]
    search_keywords = state.get("search_keywords", [query])

    if len(search_keywords) <= 1:
        # 단일 키워드 검색
        search_term = search_keywords[0] if search_keywords else query
        candidates = hybrid_search(search_term, n_results=10)
        results = reranker.rerank(query, candidates, top_k=5)
    else:
        # 복수 키워드: 각각 검색 후 reranker로 합산
        all_candidates = []
        seen_ids = set()
        for kw in search_keywords:
            candidates = hybrid_search(kw, n_results=6)
            for c in candidates:
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    all_candidates.append(c)
        results = reranker.rerank(query, all_candidates, top_k=5)

    return {
        "drug_results": results,
        "agent_trace": state.get("agent_trace", []) + ["drug_search"],
    }
