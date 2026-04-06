"""Safety Agent - 복용 안전성 확인 (임부금기, 특정연령대금기)."""

import chromadb

from src.config.settings import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_SAFETY,
)
from src.config.prompts import SAFETY_SYSTEM_PROMPT
from src.graph.state import MedAgentState
from src.vectorstore.triton_embedder import TritonEmbedder


embedder = TritonEmbedder()


def search_safety_info(query: str, n_results: int = 5) -> list[dict]:
    """ChromaDB safety 컬렉션에서 금기 정보를 검색."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        collection = client.get_collection(CHROMA_COLLECTION_SAFETY)
    except Exception:
        return []

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
            "similarity": 1 - dist,
        })

    return items


def safety_node(state: MedAgentState) -> dict:
    """Safety Agent 노드 함수."""
    query = state["query"]
    drug_results = state.get("drug_results", [])

    # 질의 + 약물명으로 safety 검색
    search_queries = [query]
    for r in drug_results[:2]:
        name = r.get("metadata", {}).get("item_name", "")
        if name:
            search_queries.append(f"{name} 금기 주의사항")

    all_results = []
    seen_ids = set()
    for sq in search_queries:
        results = search_safety_info(sq)
        for r in results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                all_results.append(r)

    # 유사도 기준 상위 5개
    all_results.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "safety_results": all_results[:5],
        "agent_trace": state.get("agent_trace", []) + ["safety"],
    }
