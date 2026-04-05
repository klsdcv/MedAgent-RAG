"""LangGraph 기반 Multi-Agent 워크플로 정의."""

from langgraph.graph import StateGraph, END

from src.graph.state import MedAgentState
from src.agents.supervisor import supervisor_node, route_by_query_type
from src.agents.drug_search import drug_search_node
from src.agents.interaction import interaction_node
from src.agents.safety import safety_node
from src.agents.answer import answer_node


def build_graph() -> StateGraph:
    """MedAgent RAG 워크플로 그래프를 생성."""
    graph = StateGraph(MedAgentState)

    # 노드 등록
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("drug_search", drug_search_node)
    graph.add_node("interaction", interaction_node)
    graph.add_node("safety", safety_node)
    graph.add_node("answer", answer_node)

    # 엣지 설정
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_by_query_type)

    # 각 Agent 실행 후 supervisor로 돌아와 다음 단계 결정
    graph.add_edge("drug_search", "supervisor")
    graph.add_edge("interaction", "supervisor")
    graph.add_edge("safety", "supervisor")
    graph.add_edge("answer", END)

    return graph


def create_app():
    """컴파일된 워크플로 앱을 반환."""
    graph = build_graph()
    return graph.compile()


def run_query(query: str) -> dict:
    """질의를 실행하고 결과를 반환."""
    app = create_app()

    initial_state = {
        "query": query,
        "query_type": "",
        "drug_results": [],
        "interaction_results": [],
        "safety_results": [],
        "final_answer": "",
        "citations": [],
        "agent_trace": [],
    }

    result = app.invoke(initial_state)
    return result
