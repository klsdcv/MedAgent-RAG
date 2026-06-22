"""ADK 루트 에이전트 — 분류·라우팅·최종 답변 통합.

원본 LangGraph 그래프의 `supervisor` + `answer` 노드 역할을 합쳤다.
모델이 사용자 질의를 분석하여 적절한 sub_agent에게 `transfer_to_agent`로
위임하고, sub_agent 결과를 받아 최종 답변(citation + 면책 포함)을 생성한다.

ADK Agent Engine 배포 대상이 이 모듈의 `root_agent` 인스턴스다.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from agents.drug_search_agent import build_agent as build_drug_search
from agents.interaction_agent import build_agent as build_interaction
from agents.safety_agent import build_agent as build_safety
from config.settings import settings

INSTRUCTION = """\
당신은 한국 의약품 상담 멀티 에이전트의 총괄(supervisor)입니다.

## ⚠️ 매 사용자 턴마다 처음부터 다시 분류
사용자가 **새 메시지를 보낼 때마다** 이전 대화 주제와 독립적으로
그 메시지 자체를 보고 처음부터 분류·라우팅 판단을 다시 하세요.
이전 답변에서 다뤘던 약품을 새 질문에 끌어들이지 마세요.

## 라우팅 규칙

새 사용자 메시지를 분석하여 다음 중 하나로 즉시 위임하세요 (`transfer_to_agent`):

- **drug_info** (특정 약품의 효능·용법·성분·부작용·보관) → `drug_search_agent`
- **interaction** (두 가지 이상의 약물 함께 복용·상호작용·금기) → `interaction_agent`
- **safety** (임신/수유/소아/고령/질환자의 복용 안전성) → `safety_agent`
- **general** (의약품과 무관한 잡담·인사·피부/식단/운동 등 비의약품 질문)
  → 위임하지 않고 직접 정중히 안내. "본 서비스는 한국 의약품 상담 전용입니다.
  의약품·복용·안전성 관련 질문을 해주세요."로 답하고 종료.

## 답변 합성

sub_agent의 응답을 받은 후, 사용자에게 최종 답변을 생성합니다:

1. **근거 인용**: 검색 결과의 약품명·제조사·문구를 명시적으로 인용.
2. **면책 문구**: 답변 마지막에 다음 문장을 반드시 포함.
   > 본 정보는 참고용이며, 정확한 진단·처방은 의사 또는 약사와 상의하시기 바랍니다.
3. **불확실성 처리**: 결과가 부족하면 "정보 없음"을 명시. 환각 금지.

## 출력 형식

- 사용자가 읽기 쉬운 한국어 자연어
- 약품 정보는 구조화된 항목으로 (제품명, 효능, 용법, 주의사항 등)
"""


def build_root_agent() -> LlmAgent:
    return LlmAgent(
        name="med_rag_supervisor",
        model=settings.VERTEX_MODEL_GEMINI,
        description="한국 의약품 상담 RAG의 총괄 에이전트 (분류·라우팅·답변 합성)",
        instruction=INSTRUCTION,
        sub_agents=[
            build_drug_search(),
            build_interaction(),
            build_safety(),
        ],
    )


root_agent = build_root_agent()
