"""약물 상호작용 확인 에이전트.

원본 LangGraph `interaction` 노드 대응. FDA OpenFDA Drug Event API + 약품
데이터스토어의 intrcQesitm 본문을 함께 활용.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from config.settings import settings
from tools.fda_dur import lookup_interaction
from tools.vertex_search import search_drugs

INSTRUCTION = """\
당신은 약물 상호작용 확인 전문 에이전트입니다.

## ⚠️ 라우팅 우선 규칙
새 메시지가 약물 상호작용·금기 주제가 아니면(예: 단일 약품 정보, 안전성, 일반 질문)
답변하지 말고 즉시 `transfer_to_agent('med_rag_supervisor')`로 부모에게 돌려보내세요.



워크플로:
1. 사용자가 언급한 두 가지 이상의 약물명을 식별합니다.
2. `search_drugs`로 한국 데이터스토어에서 각 약품의 '상호작용' 정보를 조회합니다.
3. `lookup_interaction(drug_a, drug_b)`로 FDA 사례 데이터도 함께 조회합니다.
4. 두 소스를 비교하여 다음 형식으로 보고합니다:
   - 한국 e약은요 기재 사항 (있으면)
   - FDA 사례 빈도 (results 개수)
   - 종합 권고 (의사·약사 상담 권고 포함)

면책: 의학적 판단은 항상 의료 전문가와 상의해야 한다는 점을 명시할 것.
"""


def build_agent() -> LlmAgent:
    return LlmAgent(
        name="interaction_agent",
        model=settings.VERTEX_MODEL_GEMINI,
        description="두 가지 이상의 약물 상호작용·금기를 확인하는 에이전트",
        instruction=INSTRUCTION,
        tools=[search_drugs, lookup_interaction],
    )
