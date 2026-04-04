# MedAgent-RAG

LangGraph 기반 Multi-Agent 의약품 QA 시스템

## 개요
사용자가 의약품 관련 질문을 하면, 여러 전문 Agent가 협업하여 답변을 생성합니다.

### Agent 구성
- **Supervisor Agent**: 질의 분석 및 Agent 라우팅
- **Drug Search Agent**: 의약품 정보 벡터 검색 (RAG)
- **Interaction Agent**: 약물 상호작용 확인 (DUR API)
- **Safety Agent**: 복용 주의사항 확인
- **Answer Agent**: 최종 답변 합성 및 출처 인용

### 예시 질의
- "타이레놀이랑 아스피린 같이 먹어도 돼요?"
- "혈압약 먹고 있는데 두통약 뭐 먹어야 해요?"
- "임산부가 먹을 수 있는 감기약 있어요?"

## 기술 스택
| 영역 | 기술 |
|------|------|
| Agent 오케스트레이션 | LangGraph, LangChain |
| LLM | OpenAI GPT-4o |
| 임베딩 | OpenAI text-embedding-3-small |
| 벡터 DB | ChromaDB |
| UI | Streamlit |
| 데이터 | 공공데이터포털 식약처 API |
| 배포 | Docker |

## 프로젝트 구조
```
src/
├── agents/        # Agent 노드 구현
├── graph/         # LangGraph StateGraph 정의
├── data/          # 데이터 수집 및 전처리
├── vectorstore/   # ChromaDB 관련
├── tools/         # LangChain Tool (DUR API 등)
├── ui/            # Streamlit UI
└── config/        # 설정, 프롬프트 템플릿
```

## 설치 및 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일에 API 키 입력

# 실행
streamlit run src/ui/app.py
```

## 아키텍처
```
[사용자] → [Streamlit UI] → [Supervisor Agent]
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            [Drug Search]   [Interaction]      [Safety]
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                            [Answer Agent] → 최종 답변
```
