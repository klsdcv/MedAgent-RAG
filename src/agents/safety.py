"""Safety Agent - 복용 안전성 확인 (임부금기, 특정연령대금기)."""

from src.graph.state import MedAgentState
from src.vectorstore.opensearch_client import search_safety


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
        results = search_safety(sq, n_results=5)
        for r in results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                all_results.append(r)

    # BM25 스코어 기준 상위 5개
    all_results.sort(key=lambda x: x["bm25_score"], reverse=True)

    return {
        "safety_results": all_results[:5],
        "agent_trace": state.get("agent_trace", []) + ["safety"],
    }
