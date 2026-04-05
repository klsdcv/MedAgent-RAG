"""Interaction Agent - 약물 상호작용 확인."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config.settings import OPENAI_API_KEY, OPENAI_MODEL
from src.config.prompts import INTERACTION_SYSTEM_PROMPT
from src.graph.state import MedAgentState
from src.tools.dur_api import check_drug_interaction, search_dur_by_ingredient


llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0,
).bind_tools([check_drug_interaction, search_dur_by_ingredient])


def extract_drug_names(query: str, drug_results: list) -> list[str]:
    """질의와 검색 결과에서 약물명을 추출."""
    names = []

    # 검색 결과에서 약물명 추출
    for r in drug_results[:3]:
        name = r.get("metadata", {}).get("item_name", "")
        if name:
            names.append(name)

    return names


def interaction_node(state: MedAgentState) -> dict:
    """Interaction Agent 노드 함수."""
    query = state["query"]
    drug_results = state.get("drug_results", [])
    drug_names = extract_drug_names(query, drug_results)

    # DUR API로 상호작용 확인
    interaction_results = []

    if len(drug_names) >= 2:
        # 두 약물 간 병용금기 확인
        result = check_drug_interaction.invoke({
            "drug_name_a": drug_names[0],
            "drug_name_b": drug_names[1],
        })
        interaction_results.append({"type": "병용금기", "result": result})
    elif len(drug_names) == 1:
        # 단일 약물의 DUR 정보 조회
        result = search_dur_by_ingredient.invoke({
            "ingredient_name": drug_names[0],
        })
        interaction_results.append({"type": "DUR조회", "result": result})

    return {
        "interaction_results": interaction_results,
        "agent_trace": state.get("agent_trace", []) + ["interaction"],
    }
