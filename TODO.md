# MedAgent-RAG TODO

레벨 1은 완료. 레벨 2 완료. 레벨 3 남은 작업 목록.

---

## 레벨 2

### 1. Corrective RAG (CRAG)

**무엇을:** Drug Search 결과를 LLM이 평가 → 불충분하면 쿼리 재작성 후 재검색

**작업 목록:**

- [x] `src/agents/grader.py` 생성
  - `grade_documents(query, docs) -> "relevant" | "irrelevant" | "partial"` 함수
  - GPT-4o에 질의 + 검색결과 넘겨서 관련성 판정
- [x] `src/agents/query_rewriter.py` 생성
  - `rewrite_query(original_query, reason) -> str` 함수
  - 판정이 irrelevant일 때 검색에 최적화된 쿼리 재작성 (성분명, 금기 키워드 추가 등)
- [x] `src/graph/workflow.py` 수정
  - `drug_search → grader → (relevant) → supervisor` 흐름 추가
  - `grader → (irrelevant) → query_rewriter → drug_search` 재검색 루프 추가
  - 무한루프 방지: 재검색 최대 2회 제한 (`MedAgentState`에 `search_attempts: int` 필드 추가)
- [x] `src/graph/state.py` 수정
  - `search_attempts: int` 필드 추가
  - `rewritten_query: str` 필드 추가

---

### 2. FastAPI + Streamlit 분리

**무엇을:** 지금 Streamlit이 워크플로 직접 호출하는 구조 → FastAPI 백엔드 API 서버 분리

**작업 목록:**

- [x] `src/api/main.py` 생성 (FastAPI 앱)
- [x] `src/api/routes/query.py` 생성
  - `POST /v1/query` — 일반 질의 엔드포인트
  - `POST /v1/query/stream` — SSE 스트리밍 엔드포인트 (`StreamingResponse`)
  - `GET /v1/session/{thread_id}` — 세션 상태 조회
- [x] `src/api/schemas.py` 생성
  - `QueryRequest`, `QueryResponse` Pydantic 모델 정의
- [x] `src/ui/app.py` 수정
  - `run_query()` 직접 호출 → FastAPI HTTP 호출로 교체
  - 스트리밍은 SSE (`requests` 또는 `httpx`) 로 처리
- [x] `docker/docker-compose.yml` 수정
  - `fastapi` 서비스 추가 (uvicorn)
  - `streamlit` 서비스 추가
  - 서비스 간 네트워크 설정

---

### 3. Query Rewriting Agent

**무엇을:** 사용자 구어체 질의를 검색 최적화 쿼리로 변환하는 전처리 단계

> CRAG의 재작성과 다름 — CRAG는 실패 후 교정, 이건 처음부터 최적화

**작업 목록:**

- [x] `src/agents/query_rewriter.py`에 `preprocess_query()` 함수 추가
  - "타이레놀이랑 아스피린 같이 먹어도 돼?" → "아세트아미노펜 아스피린 병용금기 상호작용"
  - 성분명 추출, 의학 용어 정규화, 검색 키워드 강조
- [x] `src/graph/workflow.py` 수정
  - Supervisor 앞에 `query_rewrite` 노드 추가
  - `entry → query_rewrite → supervisor` 로 변경
- [x] `src/graph/state.py` 수정
  - `original_query: str` 필드 추가 (원본 보존)
  - `query`는 재작성된 버전으로 사용

---

## 레벨 3

### 4. 실시간 식약처 API 연동

**무엇을:** ChromaDB 데이터가 수집 시점 스냅샷이라 최신 정보 반영 안 됨 → 실시간 API로 보완

**작업 목록:**

- [ ] `src/tools/drug_info_api.py` 생성
  - 식약처 e약은요 API 실시간 호출 LangChain Tool 구현
  - 허가 취소/변경된 약물 정보 조회
- [ ] `src/agents/drug_search.py` 수정
  - ChromaDB 검색 결과가 부족하거나 오래된 경우 실시간 API 보완 호출
- [ ] `.env` 수정
  - `DATA_API_KEY` 이미 있음, 엔드포인트 URL 상수 정리

---

### 5. Redis 캐싱

**무엇을:** 동일 질의 반복 시 LLM + 검색 재실행 방지

**작업 목록:**

- [x] `docker/docker-compose.yml`에 `redis` 서비스 추가
- [x] `src/cache/redis_client.py` 생성
  - 질의 → `sha256` 해시 키 생성
  - TTL: 24시간 (의약품 정보 변동 적음)
- [x] `src/graph/workflow.py` 수정
  - `run_query()` 앞뒤에 캐시 조회/저장 래핑
- [x] `requirements.txt`에 `redis` 추가

---

### 6. Reranker GPU 오프로드 (Triton)

**무엇을:** BGE-Reranker-v2-M3가 현재 CPU PyTorch 추론 → ONNX 변환 후 Triton에 올려서 GPU 추론

**작업 목록:**

- [x] `scripts/convert_reranker_onnx.py` 생성
  - BGE-Reranker-v2-M3 → ONNX 변환 스크립트
- [x] `triton_models/bge_reranker/` 디렉토리 구성
  - `config.pbtxt` 작성, ONNX 모델 배치
- [x] `src/vectorstore/reranker.py` 수정
  - Triton HTTP API 호출 방식으로 변경
  - Triton 미연결 시 기존 CPU fallback 유지

---

### 7. LangSmith 모니터링

**무엇을:** Agent 실행 트레이싱, 지연시간/토큰 사용량 대시보드

**작업 목록:**

- [ ] LangSmith 계정 생성 및 API 키 발급
- [ ] `.env`에 `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2=true` 추가
- [ ] `src/config/settings.py` 수정
  - LangSmith 환경변수 로드
- [ ] LangSmith 대시보드에서 확인할 항목 정리
  - 노드별 실행 시간
  - 토큰 사용량 (GPT-4o 비용 추적)
  - CRAG 재검색 발생 빈도

---

### 8. grpc -> 생각중

## 참고: 현재 완료된 것 (레벨 1)

- [x] BGE-M3 Triton 임베딩 서빙 (sentence-transformers fallback 포함)
- [x] Hybrid Search (ChromaDB Vector + OpenSearch BM25 + RRF)
- [x] BGE-Reranker-v2-M3 Cross-Encoder 재랭킹
- [x] LangGraph MemorySaver 멀티턴 대화
- [x] LangGraph stream() 기반 스트리밍 UI
- [x] RAGAS 평가 파이프라인 (faithfulness / answer_relevancy / context_precision / context_recall)
- [x] Streamlit UI (Agent 실행 흐름 사이드바, 예시 질문, 세션 관리)
