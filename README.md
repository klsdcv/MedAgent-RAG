# Vertex MedAgent RAG

원본 [MedAgent-RAG](https://github.com/klsdcv/MedAgent-RAG) (LangGraph + OpenSearch + Chroma + Triton)를 **Google Vertex AI** 스택으로 전면 재작성한 한국어 의약품 상담 RAG.

**라이브 데모:** https://med-rag-ui-719526263781.us-central1.run.app
**데이터:** 공공데이터포털 e약은요 일반의약품 4,782건

---

## 1. 아키텍처

```mermaid
flowchart LR
  USER([사용자])
  UI[Streamlit UI<br/>Cloud Run]
  AE[Agent Engine<br/>med_rag_supervisor]
  DS[drug_search_agent]
  IA[interaction_agent]
  SA[safety_agent]
  VS[(Vertex AI Search<br/>4,782 docs)]
  VEC[(Vector Search<br/>768d × 4,782)]
  FDA[FDA OpenFDA API]
  GEMINI[Gemini 2.5 Flash]
  EMB[text-multilingual-<br/>embedding-002]

  USER -->|HTTPS| UI
  UI -->|stream_query| AE
  AE -.transfer.-> DS
  AE -.transfer.-> IA
  AE -.transfer.-> SA
  AE -.LLM.-> GEMINI
  DS -->|search_drugs| VS
  DS -.LLM.-> GEMINI
  IA -->|search_drugs| VS
  IA -->|lookup_interaction| FDA
  IA -.LLM.-> GEMINI
  SA -->|search_drugs| VS
  SA -.LLM.-> GEMINI
  VEC -->|future hybrid| DS
  EMB -.embed.-> VEC
```

### 데이터 흐름
1. **사용자 질의** → Streamlit UI → Agent Engine `stream_query`
2. **Supervisor LlmAgent** (Gemini 2.5 Flash)가 질의 유형 분류 → `transfer_to_agent`로 sub-agent 위임
3. **Sub-agent**가 도구 호출 (Vertex AI Search / FDA / Vector Search)
4. **Supervisor**가 결과 받아 면책 포함 최종 답변 합성
5. **Streamlit**이 스트리밍 출력

### 인덱싱 파이프라인 (오프라인)
```mermaid
flowchart LR
  API[e약은요<br/>공공데이터 API]
  RAW[gs://...drugs_raw.json]
  PROC[gs://...drugs.jsonl]
  AISEARCH[(Vertex AI Search)]
  EMB2[text-multilingual-<br/>embedding-002]
  VEC2[(Vector Search Index)]
  EP[Index Endpoint<br/>med_rag_v1]

  API -->|collect_drugs.py| RAW
  RAW -->|preprocess.py| PROC
  PROC -->|load_to_search.py| AISEARCH
  PROC -->|load_to_vector.py| EMB2
  EMB2 -->|embeddings.json| VEC2
  VEC2 -->|deploy| EP
```

---

## 2. 원본 ↔ Vertex 매핑

| 원본 구성 | Vertex 대체 |
| --- | --- |
| LangGraph `StateGraph` (7노드) | **Google ADK** `LlmAgent` + sub_agents 패턴 (4 agent) |
| OpenSearch (sparse 키워드) | **Vertex AI Search** (Discovery Engine custom data store) |
| Chroma (dense 벡터) | **Vector Search** + `text-multilingual-embedding-002` (768d, COSINE) |
| Triton BGE-M3 임베딩/리랭커 | Vertex 임베딩 API (1차 리랭커 생략) |
| OpenAI GPT-4o | **Gemini 2.5 Flash** (Vertex) |
| Redis 캐시 | 1차 미적용 (Agent Engine session 활용) |
| FastAPI | **Agent Engine** (managed reasoning engine) |
| Streamlit | **Cloud Run** (동일 유지) |
| Triton DUR ONNX | 원본도 외부 FDA API 사용 → 그대로 |

### LangGraph 7-노드 → ADK 4-agent 통합
- 원본 `supervisor` + `answer` → **`med_rag_supervisor` LlmAgent** (분류·라우팅·최종 합성)
- 원본 `drug_search` + `grader` + `crag_rewriter` → **`drug_search_agent`** (모델 자체 판단으로 재검색 루프)
- 원본 `interaction` → **`interaction_agent`** (e약은요 + FDA 양쪽 조회)
- 원본 `safety` → **`safety_agent`**

---

## 3. 기술 스택

| 영역 | 기술 |
| --- | --- |
| 멀티에이전트 프레임워크 | Google ADK (`google-adk` ≥ 0.5) |
| LLM | Gemini 2.5 Flash (Vertex AI) |
| 임베딩 | `text-multilingual-embedding-002` (768d) |
| 키워드+시맨틱 검색 | Vertex AI Search (Discovery Engine) |
| 벡터 검색 | Vertex Vector Search (TreeAH, COSINE) |
| 에이전트 호스팅 | Vertex AI Agent Engine (managed) |
| UI 호스팅 | Cloud Run + Buildpacks |
| 스토리지 | GCS (raw / processed / embeddings) |
| 데이터 소스 | 공공데이터포털 e약은요 |
| 외부 API | OpenFDA Drug API |
| 언어/런타임 | Python 3.11 |
| 빌드 시스템 | hatchling (pyproject.toml) |

---

## 4. 디렉토리 구조

```
Vertex/
├── pyproject.toml
├── Dockerfile                      # Cloud Run Streamlit UI
├── env.example
├── deploy/
│   ├── 00_enable_apis.sh           # 9개 GCP API 활성화
│   ├── 01_create_buckets.sh        # GCS staging + data 버킷
│   ├── 03_build_vector_index.py    # 인덱스 + 엔드포인트 + deploy
│   ├── 04_deploy_agent_engine.py   # ADK 루트 에이전트 배포
│   └── 05_deploy_ui_cloudrun.sh    # Streamlit Cloud Run 배포
├── config/
│   ├── settings.py                 # pydantic-settings (env 자동 로드)
│   └── prompts.py
├── agents/
│   ├── root_agent.py               # supervisor (분류·합성)
│   ├── drug_search_agent.py
│   ├── interaction_agent.py
│   └── safety_agent.py
├── tools/
│   ├── vertex_search.py            # Discovery Engine Search 호출
│   ├── vector_search.py            # 임베딩 → KNN → 메타데이터 결합
│   └── fda_dur.py                  # OpenFDA Drug Event API
├── ingest/
│   ├── collect_drugs.py            # e약은요 → GCS raw
│   ├── preprocess.py               # HTML 제거, JSONL 변환
│   ├── load_to_search.py           # → Vertex AI Search import
│   └── load_to_vector.py           # → 임베딩 → Vector Search 입력
├── ui/
│   └── app.py                      # Streamlit (Agent Engine stream_query)
├── tests/
│   └── test_local_agent.py         # InMemoryRunner 로컬 검증
└── _reference/MedAgent-RAG/        # 원본 참조용 (gitignore)
```

---

## 5. 처음부터 재현하기

```bash
# 0. Python 3.11 + venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ingest]"
cp env.example .env  # 값 채우기 (DATA_API_KEY 필수)

# 1. GCP 사전 준비
gcloud auth application-default login
bash deploy/00_enable_apis.sh
bash deploy/01_create_buckets.sh

# 2. 데이터 인제스천
python -m ingest.collect_drugs        # 4799건 → GCS
python -m ingest.preprocess           # HTML 제거 → JSONL (4782건)
python -m ingest.load_to_search       # Vertex AI Search import
python -m ingest.load_to_vector       # 임베딩 (4분 40초)

# 3. Vector Search 인덱스
python deploy/03_build_vector_index.py   # 30-50분 (deploy 포함)
# → VECTOR_INDEX_ID, VECTOR_INDEX_ENDPOINT_ID 출력. .env에 채움

# 4. IAM (Agent Engine SA에 권한 부여)
for SA in 719526263781-compute@developer.gserviceaccount.com \
          service-719526263781@gcp-sa-aiplatform.iam.gserviceaccount.com \
          service-719526263781@gcp-sa-aiplatform-re.iam.gserviceaccount.com; do
  for R in discoveryengine.viewer aiplatform.user storage.objectViewer; do
    gcloud projects add-iam-policy-binding civic-athlete-500200-t0 \
      --member="serviceAccount:${SA}" --role="roles/${R}" --condition=None
  done
done

# 5. ADK 에이전트 로컬 검증 → Agent Engine 배포
python -m tests.test_local_agent "활명수 효능 알려줘"
python deploy/04_deploy_agent_engine.py
# → AGENT_ENGINE_RESOURCE_NAME 출력. .env에 채움

# 6. Streamlit UI Cloud Run 배포
bash deploy/05_deploy_ui_cloudrun.sh
# → https://med-rag-ui-...run.app 출력
```

---

## 6. 운영 비용 추정

| 리소스 | 단가 | 비고 |
| --- | --- | --- |
| Vector Search 엔드포인트 | ~$0.07/시간 (e2-standard-2) | 살아있는 동안 계속 |
| Agent Engine | invoke당 과금 (Gemini 토큰) | idle 시 0 |
| Cloud Run UI | 요청당 과금 + 최소 인스턴스 0 | 거의 0 |
| Vertex AI Search | 데이터스토어 idle 시 0 | 검색당 미세 과금 |
| GCS | < 100MB | 거의 0 |

데모만 잠깐 쓰면 월 $5 이내, Vector Search 엔드포인트를 계속 켜두면 월 ~$50. 사용 안 할 때 endpoint를 `undeploy_index()`하면 비용 0.

---

## 7. 배운 점 / 트러블슈팅

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| 인덱스 생성 `FAILED_PRECONDITION` | `.jsonl` 확장자 미인식 | 파일명을 `.json`으로 |
| 인덱스 생성 시 restricts 의심 | 한국어 allow 값 (회사명) | restricts 제거 후 성공 (별개 fix) |
| 임베딩 토큰 한도 초과 | 한국어는 1.4 char/token | 동적 배치 (글자수 누적 22000자 기준) |
| Agent Engine 배포 후 컨테이너 fail | `Settings` 필수 필드 missing | `env_vars=` 인자로 GCP/Vertex 환경 주입 |
| Agent Engine `discoveryengine.search` 403 | P4SA에 권한 없음 | 3개 SA × 3개 role 부여 |

---

## 8. 한계 및 향후 개선

- **CRAG grader + rewriter 루프**가 ADK의 sub_agent 자체 판단으로 단순화 → 명시적 `LoopAgent` 도입 가능
- **하이브리드 검색** 통합 (Vertex AI Search + Vector Search를 RRF로 합치는 단계 추가)
- **리랭커** (Vertex Ranking API) 추가
- **세션 이력 영속화** (Agent Engine SessionService → Firestore)
- **응답 캐시** (Memorystore Redis)
- **평가 자동화** (원본의 eval 데이터셋 포팅)
