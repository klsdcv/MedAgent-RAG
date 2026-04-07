"""Query Rewriter Agent - 쿼리 전처리 + CRAG 재작성."""

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config.settings import OPENAI_API_KEY, OPENAI_MODEL
from src.config.prompts import QUERY_REWRITE_SYSTEM_PROMPT
from src.graph.state import MedAgentState


llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0,
)

# ── 전처리용 프롬프트 (entry 단계) ──────────────────────────────────────────────

PREPROCESS_PROMPT = """사용자의 구어체 질의를 의약품 검색에 최적화된 쿼리로 변환하세요.

변환 규칙:
1. 상품명이 있으면 유지하되, 성분명도 병기
2. 구어체 표현을 의학/약학 검색 용어로 변환
3. 검색에 불필요한 조사, 어미 제거
4. 핵심 의약 키워드 강조

예시:
- "타이레놀이랑 아스피린 같이 먹어도 돼?" → "타이레놀 아세트아미노펜 아스피린 병용금기 상호작용"
- "임산부가 먹을 수 있는 감기약?" → "임산부 감기약 해열진통제 항히스타민 임부금기"
- "혈압약 먹는데 두통약 뭐가 좋아?" → "항고혈압제 혈압강하제 두통 해열진통제 병용"
- "속쓰림에 먹는 약 추천" → "위산과다 속쓰림 제산제 위장약 프로톤펌프억제제"

반드시 아래 형식의 JSON만 답하세요:
{{"rewritten": "변환된 쿼리 문자열"}}

사용자 질의: {query}"""

# ── CRAG 재작성용 프롬프트 (검색 실패 후) ────────────────────────────────────────

CRAG_REWRITE_PROMPT = """이전 검색이 관련성 낮은 결과를 반환했습니다. 검색 쿼리를 재작성하세요.

원본 질의: {original_query}
이전 검색 쿼리: {previous_query}
실패 사유: {reason}

재작성 전략:
1. 성분명, 약효 분류명 등 구체적 의학 용어 추가
2. 동의어/유사어 확장 (예: 진통제 → 해열진통제, NSAIDs, 아세트아미노펜)
3. 검색 범위를 넓히되 관련성 유지
4. 이전 쿼리와 다른 접근으로 재작성

반드시 아래 형식의 JSON만 답하세요:
{{"rewritten": "재작성된 검색 쿼리"}}"""


def _parse_rewrite_response(content: str, fallback: str) -> str:
    """LLM 응답에서 재작성된 쿼리를 파싱."""
    content = content.strip()
    try:
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
        result = parsed.get("rewritten", fallback)
        return result if result else fallback
    except (json.JSONDecodeError, IndexError):
        return fallback


def preprocess_query(query: str) -> str:
    """사용자 구어체 질의를 검색 최적화 쿼리로 전처리.

    CRAG 재작성과 다름 — 이 함수는 최초 진입 시 1회 실행.
    """
    response = llm.invoke([
        SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=PREPROCESS_PROMPT.format(query=query)),
    ])
    return _parse_rewrite_response(response.content, fallback=query)


def rewrite_query(original_query: str, previous_query: str, reason: str = "") -> str:
    """CRAG 실패 후 검색 쿼리 재작성.

    Args:
        original_query: 사용자 원본 질의
        previous_query: 이전에 사용한 검색 쿼리
        reason: Grader가 제공한 실패 사유

    Returns:
        재작성된 검색 쿼리
    """
    response = llm.invoke([
        SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=CRAG_REWRITE_PROMPT.format(
            original_query=original_query,
            previous_query=previous_query,
            reason=reason,
        )),
    ])
    return _parse_rewrite_response(response.content, fallback=original_query)


def query_rewrite_node(state: MedAgentState) -> dict:
    """Query Rewrite 노드 — 진입 시 전처리."""
    query = state["query"]
    rewritten = preprocess_query(query)

    return {
        "original_query": query,
        "query": rewritten,
        "agent_trace": state.get("agent_trace", []) + ["query_rewrite"],
    }


def crag_rewrite_node(state: MedAgentState) -> dict:
    """CRAG Rewrite 노드 — 검색 실패 후 쿼리 재작성."""
    original = state.get("original_query", "") or state["query"]
    previous = state["query"]

    rewritten = rewrite_query(original, previous)

    return {
        "query": rewritten,
        "rewritten_query": rewritten,
        "drug_results": [],  # 이전 결과 초기화
        "agent_trace": state.get("agent_trace", []) + ["crag_rewrite"],
    }
