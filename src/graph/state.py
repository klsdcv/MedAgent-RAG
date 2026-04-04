from typing import TypedDict


class MedAgentState(TypedDict):
    query: str                  # 사용자 질의
    query_type: str             # 분류 결과 (simple, interaction, safety, complex)
    drug_results: list          # Drug Search Agent 결과
    interaction_results: list   # Interaction Agent 결과
    safety_results: list        # Safety Agent 결과
    final_answer: str           # 최종 답변
    citations: list             # 출처 목록
    agent_trace: list           # 호출된 Agent 이력
