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

CLASSIFY_PROMPT = """사용자 질의를 분석하여 다음 두 가지를 JSON으로 답하세요.

1. query_type: 아래 4가지 유형 중 하나
   - simple: 단순 약 정보 조회 (예: "타이레놀 효능 알려줘")
   - interaction: 약물 상호작용 확인 (예: "타이레놀이랑 아스피린 같이 먹어도 돼?")
   - safety: 복용 주의사항 확인 (예: "임산부가 먹을 수 있는 감기약?")
   - complex: 복합 질의 (예: "혈압약 먹고 있는데 두통약 추천해줘")

2. search_keywords: 의약품 데이터베이스 검색에 사용할 키워드 리스트
   - 구어체를 검색에 적합한 의학/약학 용어로 변환
   - 제품명이 있으면 그대로 유지
   - 일반 약 종류면 관련 성분명이나 약효 분류를 추가
   - 예: "혈압약 먹고 있는데 두통약 추천해줘" → ["혈압 강압제 항고혈압", "두통 해열진통제 아세트아미노펜"]
   - 예: "타이레놀 효능 알려줘" → ["타이레놀"]
   - 예: "관절약이랑 소화제 같이 먹어도 돼?" → ["관절 글루코사민 소염진통", "소화제 소화효소"]

반드시 아래 형식의 JSON만 답하세요:
{{"query_type": "...", "search_keywords": [...]}}

사용자 질의: {query}"""


def classify_query(query: str) -> tuple[str, list[str]]:
    """질의 유형 분류 + 검색 키워드 추출."""
    import json

    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=CLASSIFY_PROMPT.format(query=query)),
    ])

    content = response.content.strip()

    # JSON 파싱
    try:
        # ```json ... ``` 블록 처리
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
        query_type = parsed.get("query_type", "simple").lower()
        search_keywords = parsed.get("search_keywords", [query])
    except (json.JSONDecodeError, IndexError):
        query_type = "simple"
        search_keywords = [query]

    valid_types = {"simple", "interaction", "safety", "complex"}
    if query_type not in valid_types:
        query_type = "simple"

    if not search_keywords:
        search_keywords = [query]

    return query_type, search_keywords


def supervisor_node(state: MedAgentState) -> dict:
    """Supervisor Agent 노드 함수."""
    query = state["query"]
    query_type, search_keywords = classify_query(query)

    return {
        "query_type": query_type,
        "search_keywords": search_keywords,
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
