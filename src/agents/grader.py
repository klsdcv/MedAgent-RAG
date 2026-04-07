"""Grader Agent - 검색 결과 관련성 평가 (Corrective RAG)."""

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config.settings import OPENAI_API_KEY, OPENAI_MODEL
from src.config.prompts import GRADER_SYSTEM_PROMPT
from src.graph.state import MedAgentState


llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0,
)

GRADE_PROMPT = """사용자 질의와 검색된 문서들을 비교하여 관련성을 평가하세요.

사용자 질의: {query}

검색 결과:
{documents}

평가 기준 (관대하게 판정):
- relevant: 질의에 언급된 약물이나 성분이 검색 결과에 1건이라도 포함되면 relevant
- partial: 직접적인 약물은 없지만 관련 약효 분류나 유사 성분 정보가 있음
- irrelevant: 검색 결과가 질의와 완전히 무관 (약물명, 성분명, 약효 분류 어느 것도 일치하지 않음)

중요: 의심스러우면 relevant로 판정하세요. irrelevant는 정말 무관한 경우에만 사용합니다.

반드시 아래 형식의 JSON만 답하세요:
{{"grade": "relevant|partial|irrelevant", "reason": "판정 사유"}}"""


def grade_documents(query: str, docs: list[dict]) -> tuple[str, str]:
    """검색 결과의 관련성을 평가.

    Args:
        query: 사용자 질의
        docs: 검색 결과 리스트 (각 항목에 document, metadata 포함)

    Returns:
        (grade, reason) 튜플. grade는 "relevant", "partial", "irrelevant" 중 하나.
    """
    if not docs:
        return "irrelevant", "검색 결과 없음"

    doc_texts = []
    for i, doc in enumerate(docs[:5], 1):
        name = doc.get("metadata", {}).get("item_name", "unknown")
        text = doc.get("document", "")[:300]
        doc_texts.append(f"[{i}] {name}: {text}")
    documents_str = "\n".join(doc_texts)

    response = llm.invoke([
        SystemMessage(content=GRADER_SYSTEM_PROMPT),
        HumanMessage(content=GRADE_PROMPT.format(query=query, documents=documents_str)),
    ])

    content = response.content.strip()

    try:
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
        grade = parsed.get("grade", "relevant").lower()
        reason = parsed.get("reason", "")
    except (json.JSONDecodeError, IndexError):
        grade = "relevant"
        reason = "파싱 실패 — 기본값 relevant 적용"

    valid_grades = {"relevant", "partial", "irrelevant"}
    if grade not in valid_grades:
        grade = "relevant"

    return grade, reason


def grader_node(state: MedAgentState) -> dict:
    """Grader Agent 노드 함수.

    drug_search 결과를 평가하고, irrelevant이면 재검색 트리거.
    """
    query = state.get("original_query", "") or state["query"]
    docs = state.get("drug_results", [])
    attempts = state.get("search_attempts", 0)

    grade, reason = grade_documents(query, docs)

    update = {
        "agent_trace": state.get("agent_trace", []) + ["grader"],
    }

    if grade in ("relevant", "partial"):
        # 검색 결과 충분 또는 부분 관련 — 다음 단계로 진행
        update["search_attempts"] = attempts
    elif grade == "irrelevant" and attempts < 2:
        # 완전 무관한 경우만 재검색
        update["search_attempts"] = attempts + 1
        update["rewritten_query"] = ""  # query_rewriter가 채울 예정
    else:
        # 재검색 횟수 초과 — 현재 결과로 진행
        update["search_attempts"] = attempts

    # grade를 라우팅에서 사용하기 위해 임시 저장
    update["_grade"] = grade

    return update


def route_after_grading(state: MedAgentState) -> str:
    """Grader 결과에 따라 다음 노드 결정."""
    grade = state.get("_grade", "relevant")
    attempts = state.get("search_attempts", 0)

    if grade in ("relevant", "partial") or attempts >= 2:
        return "supervisor_route"
    else:
        return "crag_rewriter"
