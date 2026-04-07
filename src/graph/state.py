from typing import TypedDict, Annotated
import operator


class MedAgentState(TypedDict):
    query: str                  # 사용자 질의 (전처리 후 검색 최적화된 버전)
    original_query: str         # 사용자 원본 질의 (전처리 전)
    query_type: str             # 분류 결과 (simple, interaction, safety, complex)
    search_keywords: list       # Supervisor가 추출한 검색 키워드
    drug_results: list          # Drug Search Agent 결과
    interaction_results: list   # Interaction Agent 결과
    safety_results: list        # Safety Agent 결과
    final_answer: str           # 최종 답변
    citations: list             # 출처 목록
    agent_trace: list           # 호출된 Agent 이력
    messages: Annotated[list, operator.add]  # 대화 히스토리 (멀티턴)
    # CRAG 관련
    search_attempts: int        # 검색 시도 횟수 (재검색 루프 제한용, 최대 2)
    rewritten_query: str        # Grader가 irrelevant 판정 시 재작성된 쿼리
    _grade: str                 # Grader 판정 결과 (라우팅용 내부 필드)
