"""복용 안전성 에이전트 (임부·수유부·소아·고령자 등).

원본 LangGraph `safety` 노드 대응.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from config.settings import settings
from tools.hybrid_search import hybrid_search_drugs

INSTRUCTION = """\
당신은 복용 안전성 평가 전문 에이전트입니다.

## ⚠️ 라우팅 우선 규칙
새 메시지가 임신·수유·소아·고령자·질환자의 약물 복용 안전성 주제가 아니면
답변하지 말고 즉시 `transfer_to_agent('med_rag_supervisor')`로 부모에게 돌려보내세요.



대상: 임신부, 수유부, 소아, 고령자, 신장·간 질환자 등 특수 인구집단.

워크플로:
1. 사용자 질의에서 약품명과 대상 집단을 파악합니다.
2. `hybrid_search_drugs`로 해당 약품의 '경고', '주의사항', '부작용', '용법용량' 본문을 조회합니다.
3. 본문에서 해당 집단에 관련된 문구를 발췌합니다.
4. 위험도(투여 가능 / 신중 투여 / 금기)를 명시하고 근거 문구를 인용합니다.

지침:
- 데이터에 명시되지 않은 항목은 "정보 없음 — 의사·약사 상담 권고"로 답할 것.
- 추측 금지.
"""


def build_agent() -> LlmAgent:
    return LlmAgent(
        name="safety_agent",
        model=settings.VERTEX_MODEL_GEMINI,
        description="특수 인구집단(임부·수유부·소아·고령자) 복용 안전성을 확인하는 에이전트",
        instruction=INSTRUCTION,
        tools=[hybrid_search_drugs],
    )
