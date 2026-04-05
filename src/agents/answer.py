"""Answer Agent - 최종 답변 합성 및 출처 인용."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config.settings import OPENAI_API_KEY, OPENAI_MODEL
from src.config.prompts import ANSWER_SYSTEM_PROMPT
from src.graph.state import MedAgentState


llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.3,
)


def build_context(state: MedAgentState) -> str:
    """Agent 결과들을 컨텍스트 텍스트로 합침."""
    sections = []

    drug_results = state.get("drug_results", [])
    if drug_results:
        drug_texts = []
        for r in drug_results[:3]:
            drug_texts.append(f"[{r['metadata']['item_name']}]\n{r['document']}")
        sections.append("## 의약품 검색 결과\n" + "\n\n".join(drug_texts))

    interaction_results = state.get("interaction_results", [])
    if interaction_results:
        sections.append("## 약물 상호작용 정보\n" + "\n".join(
            str(r) for r in interaction_results[:5]
        ))

    safety_results = state.get("safety_results", [])
    if safety_results:
        sections.append("## 복용 안전성 정보\n" + "\n".join(
            str(r) for r in safety_results[:5]
        ))

    return "\n\n".join(sections)


def answer_node(state: MedAgentState) -> dict:
    """Answer Agent 노드 함수."""
    query = state["query"]
    context = build_context(state)

    prompt = f"""다음 정보를 바탕으로 사용자의 질문에 답변하세요.

## 사용자 질문
{query}

## 참고 정보
{context}

답변:"""

    response = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    # 출처 목록 생성
    citations = []
    for r in state.get("drug_results", [])[:3]:
        citations.append({
            "item_name": r["metadata"]["item_name"],
            "item_seq": r["metadata"]["item_seq"],
        })

    return {
        "final_answer": response.content,
        "citations": citations,
        "agent_trace": state.get("agent_trace", []) + ["answer"],
    }
