"""LangGraph 기반 Multi-Agent 워크플로 정의."""

import uuid
from collections.abc import Generator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

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


def create_app(checkpointer: MemorySaver | None = None):
    """컴파일된 워크플로 앱을 반환."""
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)


# 전역 checkpointer (메모리 기반 세션 유지)
_checkpointer = MemorySaver()
_app = create_app(checkpointer=_checkpointer)


def run_query(query: str, thread_id: str | None = None) -> dict:
    """질의를 실행하고 결과를 반환.

    Args:
        query: 사용자 질의
        thread_id: 대화 세션 ID. None이면 새 세션 생성.

    Returns:
        최종 상태 dict
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "query": query,
        "query_type": "",
        "search_keywords": [],
        "drug_results": [],
        "interaction_results": [],
        "safety_results": [],
        "final_answer": "",
        "citations": [],
        "agent_trace": [],
        "messages": [{"role": "user", "content": query}],
    }

    result = _app.invoke(initial_state, config=config)
    return result


_NODE_LABELS = {
    "supervisor": "🎯 Supervisor — 질의 유형 분석 중...",
    "drug_search": "🔍 Drug Search — 의약품 검색 중...",
    "interaction": "⚡ Interaction — 약물 상호작용 확인 중...",
    "safety": "🛡️ Safety — 복용 안전성 확인 중...",
    "answer": "✍️ Answer — 답변 생성 중...",
}


def stream_query(query: str, thread_id: str | None = None) -> Generator[str, None, dict]:
    """질의를 스트리밍으로 실행.

    노드 실행마다 상태 메시지를 yield하고,
    Answer 노드 완료 후 최종 답변 텍스트를 yield.

    Returns:
        Generator — 각 yield 값은 UI에 출력할 텍스트 조각
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "query": query,
        "query_type": "",
        "search_keywords": [],
        "drug_results": [],
        "interaction_results": [],
        "safety_results": [],
        "final_answer": "",
        "citations": [],
        "agent_trace": [],
        "messages": [{"role": "user", "content": query}],
    }

    final_state = {}
    seen_nodes = set()

    for chunk in _app.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, state_update in chunk.items():
            if node_name not in seen_nodes:
                seen_nodes.add(node_name)
                label = _NODE_LABELS.get(node_name, f"⚙️ {node_name}...")
                yield f"\n`{label}`\n"

            final_state.update(state_update)

    # 최종 답변 출력
    answer = final_state.get("final_answer", "")
    if answer:
        yield "\n\n"
        yield answer

    return final_state
