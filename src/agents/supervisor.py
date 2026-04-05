"""Supervisor Agent - 질의 분석 및 Agent 라우팅."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config.settings import OPENAI_API_KEY, OPENAI_MODEL
from src.config.prompts import SUPERVISOR_SYSTEM_PROMPT
from src.graph.state import MedAgentState


llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0,
)

CLASSIFY_PROMPT = """사용자 질의를 분석하여 아래 4가지 유형 중 하나로 분류하세요.
반드시 유형명만 답하세요.

유형:
- simple: 단순 약 정보 조회 (예: "타이레놀 효능 알려줘")
- interaction: 약물 상호작용 확인 (예: "타이레놀이랑 아스피린 같이 먹어도 돼?")
- safety: 복용 주의사항 확인 (예: "임산부가 먹을 수 있는 감기약?")
- complex: 복합 질의 (예: "혈압약 먹고 있는데 두통약 추천해줘")

사용자 질의: {query}

유형:"""


def classify_query(query: str) -> str:
    """질의 유형을 분류."""
    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=CLASSIFY_PROMPT.format(query=query)),
    ])
    query_type = response.content.strip().lower()

    valid_types = {"simple", "interaction", "safety", "complex"}
    if query_type not in valid_types:
        query_type = "simple"

    return query_type


def supervisor_node(state: MedAgentState) -> dict:
    """Supervisor Agent 노드 함수."""
    query = state["query"]
    query_type = classify_query(query)

    return {
        "query_type": query_type,
        "agent_trace": state.get("agent_trace", []) + ["supervisor"],
    }


def route_by_query_type(state: MedAgentState) -> str:
    """질의 유형에 따라 다음 Agent를 결정."""
    query_type = state.get("query_type", "simple")
    trace = state.get("agent_trace", [])

    # 이미 drug_search를 거쳤는지 확인
    has_drug_search = "drug_search" in trace
    has_interaction = "interaction" in trace
    has_safety = "safety" in trace

    if not has_drug_search:
        return "drug_search"

    if query_type == "simple":
        return "answer"
    elif query_type == "interaction":
        return "interaction" if not has_interaction else "answer"
    elif query_type == "safety":
        return "safety" if not has_safety else "answer"
    elif query_type == "complex":
        if not has_interaction:
            return "interaction"
        if not has_safety:
            return "safety"
        return "answer"

    return "answer"
