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


def build_context(state: MedAgentState) -> tuple[str, list[dict]]:
    """Agent 결과들을 번호가 붙은 컨텍스트 텍스트로 합침.

    Returns:
        (context_text, citations) 튜플.
        citations: [{"index": 1, "item_name": ..., "item_seq": ..., "source": ...}, ...]
    """
    sections = []
    citations = []
    ref_num = 1

    drug_results = state.get("drug_results", [])
    if drug_results:
        drug_texts = []
        for r in drug_results[:5]:
            name = r["metadata"]["item_name"]
            doc_preview = r["document"][:200].replace("\n", " ")
            drug_texts.append(f"[{ref_num}] {r['document']}")
            citations.append({
                "index": ref_num,
                "item_name": name,
                "item_seq": r["metadata"].get("item_seq", ""),
                "source": "의약품 정보 (e약은요)",
                "preview": doc_preview,
            })
            ref_num += 1
        sections.append("## 의약품 검색 결과\n" + "\n\n".join(drug_texts))

    interaction_results = state.get("interaction_results", [])
    if interaction_results:
        inter_texts = []
        for r in interaction_results[:5]:
            r_type = r.get("type", "") if isinstance(r, dict) else ""
            r_result = r.get("result", "") if isinstance(r, dict) else str(r)
            display = f"[{r_type}] {r_result}" if r_type else str(r_result)
            inter_texts.append(f"[{ref_num}] {display}")
            citations.append({
                "index": ref_num,
                "item_name": r_type or "약물 상호작용",
                "item_seq": "",
                "source": "DUR 병용금기 정보",
                "preview": str(r_result)[:200],
            })
            ref_num += 1
        sections.append("## 약물 상호작용 정보\n" + "\n".join(inter_texts))

    safety_results = state.get("safety_results", [])
    if safety_results:
        safety_texts = []
        for r in safety_results[:5]:
            doc = r.get("document", str(r)) if isinstance(r, dict) else str(r)
            name = r.get("metadata", {}).get("item_name", "") if isinstance(r, dict) else ""
            safety_texts.append(f"[{ref_num}] {doc}")
            citations.append({
                "index": ref_num,
                "item_name": name or doc.split("\n")[0][:50],
                "item_seq": "",
                "source": "DUR 안전성 정보",
                "preview": doc[:200],
            })
            ref_num += 1
        sections.append("## 복용 안전성 정보\n" + "\n".join(safety_texts))

    return "\n\n".join(sections), citations


def build_history_prompt(messages: list[dict]) -> str:
    """이전 대화 히스토리를 프롬프트 텍스트로 변환."""
    # 마지막 user 메시지 제외하고 최근 6턴만 포함
    history = messages[:-1][-6:] if len(messages) > 1 else []
    if not history:
        return ""

    lines = ["## 이전 대화"]
    for msg in history:
        role = "사용자" if msg["role"] == "user" else "어시스턴트"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def answer_node(state: MedAgentState) -> dict:
    """Answer Agent 노드 함수."""
    query = state["query"]
    context, citations = build_context(state)
    history = build_history_prompt(state.get("messages", []))

    history_section = f"\n{history}\n" if history else ""

    prompt = f"""다음 정보를 바탕으로 사용자의 질문에 답변하세요.
{history_section}
## 사용자 질문
{query}

## 참고 정보
{context}

답변:"""

    response = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {
        "final_answer": response.content,
        "citations": citations,
        "agent_trace": state.get("agent_trace", []) + ["answer"],
        "messages": [{"role": "assistant", "content": response.content}],
    }
