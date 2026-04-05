# MedAgent-RAG

LangGraph 기반 Multi-Agent 의약품 QA 시스템

## 개요
사용자가 의약품 관련 질문을 하면, 여러 전문 Agent가 협업하여 답변을 생성하는 Multi-Agent RAG 시스템입니다.

### 예시 질의
- "타이레놀이랑 아스피린 같이 먹어도 돼요?"
- "혈압약 먹고 있는데 두통약 뭐 먹어야 해요?"
- "임산부가 먹을 수 있는 감기약 있어요?"
- "메트포르민 부작용 알려줘"

## 아키텍처

```
[사용자] → [Streamlit UI] → [Supervisor Agent] (질의 분류 + 동적 라우팅)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            [Drug Search]   [Interaction]      [Safety]
            Hybrid Search   DUR API 실시간     ChromaDB
           (Vector + BM25)    Tool Use        금기정보 검색
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                            [Answer Agent] → GPT-4o 답변 생성 + 출처 인용
```

### Agent 상세

| Agent | 역할 | 구현 |
|-------|------|------|
| **Supervisor** | 질의 유형 분류 (simple/interaction/safety/complex) + Agent 라우팅 | LangGraph conditional_edges |
| **Drug Search** | 의약품 정보 검색 | Hybrid Search (Vector + BM25) + RRF re-ranking |
| **Interaction** | 약물 상호작용 확인 | DUR 병용금기 API 실시간 호출 (LangChain Tool Use) |
| **Safety** | 복용 주의사항 확인 (임부금기, 연령대금기) | ChromaDB safety 컬렉션 벡터 검색 |
| **Answer** | 최종 답변 합성 + 출처 인용 + 면책 문구 | GPT-4o |

### 질의 유형별 라우팅

| 질의 유형 | 호출 경로 | 예시 |
|-----------|----------|------|
| 단순 약 정보 | Supervisor → Drug Search → Answer | "타이레놀 효능 알려줘" |
| 약물 상호작용 | Supervisor → Drug Search → Interaction → Answer | "타이레놀이랑 아스피린 같이 먹어도 돼?" |
| 복용 주의 | Supervisor → Drug Search → Safety → Answer | "임산부가 먹을 수 있는 감기약?" |
| 복합 질의 | Supervisor → Drug Search → Interaction → Safety → Answer | "혈압약 먹고 있는데 두통약 추천해줘" |

## 기술 스택

| 영역 | 기술 |
|------|------|
| Agent 오케스트레이션 | LangGraph, LangChain |
| LLM | OpenAI GPT-4o |
| 임베딩 | BGE-M3 (ONNX + Triton Inference Server) |
| 검색 | Hybrid Search (ChromaDB Vector + OpenSearch BM25 + RRF) |
| 벡터 DB | ChromaDB |
| 키워드 검색 | OpenSearch 2.18 (nori 한국어 형태소 분석기) |
| 데이터 | 공공데이터포털 식약처 API (e약은요, DUR) |
| UI | Streamlit |
| 배포 | Docker Compose (Triton + OpenSearch) |

## 검색 파이프라인

### Hybrid Search (Vector + OpenSearch BM25)

```
[사용자 질의]
      │
      ├──▶ [BGE-M3 임베딩] → ChromaDB 벡터 검색 (의미 유사도)
      │
      ├──▶ [OpenSearch] → nori 한국어 형태소 분석 → BM25 키워드 검색 (정확 매칭)
      │
      └──▶ [RRF (Reciprocal Rank Fusion)] → 최종 순위 결정
```

- **벡터 검색**: "두통약 추천" → 해열진통제 계열 의약품 매칭 (의미 기반)
- **OpenSearch BM25**: "타이레놀" → 정확한 약물명 매칭 (nori 형태소 분석)
- **RRF**: 두 결과를 가중 합산하여 re-ranking (vector 60% + BM25 40%)

### 임베딩 서빙

- **모델**: BAAI/bge-m3 (1024차원, 다국어)
- **서빙**: ONNX 변환 → Triton Inference Server (CPU)
- **변환**: `scripts/convert_bge_m3_onnx.py`

## 데이터

| 데이터 | 출처 | 건수 | 용도 |
|--------|------|------|------|
| 의약품개요정보 (e약은요) | 식약처 공공데이터포털 | 4,697건 | Drug Search (효능, 용법, 성분) |
| DUR 특정연령대금기 | 식약처 DUR | 2,666건 | Safety Agent |
| DUR 임부금기 | 식약처 DUR | 16,276건 | Safety Agent |
| DUR 병용금기 | 식약처 DUR API | 실시간 호출 | Interaction Agent (Tool Use) |

## 프로젝트 구조

```
MedAgent-RAG/
├── src/
│   ├── agents/              # Agent 노드 구현
│   │   ├── supervisor.py    # 질의 분류 + 라우팅
│   │   ├── drug_search.py   # 하이브리드 검색 (Vector + OpenSearch BM25)
│   │   ├── interaction.py   # DUR API 약물 상호작용 확인
│   │   ├── safety.py        # 임부금기/연령대금기 검색
│   │   └── answer.py        # 최종 답변 생성
│   ├── graph/
│   │   ├── state.py         # MedAgentState 타입 정의
│   │   └── workflow.py      # LangGraph StateGraph 워크플로
│   ├── data/                # 데이터 수집 및 전처리
│   │   ├── collect_drugs.py # e약은요 API 수집
│   │   ├── collect_dur.py   # DUR API 수집
│   │   ├── preprocess_drugs.py
│   │   ├── preprocess_dur.py
│   │   ├── load_to_chroma.py
│   │   └── load_to_opensearch.py
│   ├── vectorstore/
│   │   ├── triton_embedder.py    # Triton HTTP 임베딩 클라이언트
│   │   └── opensearch_client.py  # OpenSearch BM25 검색 클라이언트
│   ├── tools/
│   │   └── dur_api.py       # DUR 병용금기 API (LangChain Tool)
│   ├── ui/                  # Streamlit UI
│   └── config/
│       ├── settings.py      # 환경변수, DB 설정
│       └── prompts.py       # Agent 프롬프트 템플릿
├── scripts/
│   └── convert_bge_m3_onnx.py  # BGE-M3 → ONNX 변환
├── triton_models/
│   └── bge_m3/config.pbtxt # Triton 모델 설정
├── docker/
│   ├── docker-compose.yml         # Triton + OpenSearch
│   └── opensearch.Dockerfile      # nori 플러그인 포함 이미지
├── tests/
├── requirements.txt
└── .gitignore
```

## 설치 및 실행

### 1. 환경 설정

```bash
# Python 3.11 가상환경
python3.11 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수
cp .env.example .env
# .env에 OPENAI_API_KEY, DATA_API_KEY, HF_TOKEN 입력
```

### 2. 인프라 (Triton + OpenSearch)

```bash
# BGE-M3 → ONNX 변환
python scripts/convert_bge_m3_onnx.py

# Triton + OpenSearch 실행
docker compose -f docker/docker-compose.yml up -d
```

### 3. 데이터 수집 및 적재

```bash
# e약은요 수집 → 전처리 → ChromaDB + OpenSearch 적재
python -m src.data.collect_drugs
python -m src.data.preprocess_drugs
python -m src.data.load_to_chroma
python -m src.data.load_to_opensearch

# DUR 수집 → 전처리
python -m src.data.collect_dur
python -m src.data.preprocess_dur
```

### 4. 실행

```bash
streamlit run src/ui/app.py
```
