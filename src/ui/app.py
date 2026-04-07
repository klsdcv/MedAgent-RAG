"""MedAgent-RAG Streamlit UI — FastAPI 백엔드 연동."""

import os
import json
import uuid

import streamlit as st
import httpx

# ── API 설정 ─────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAgent-RAG",
    page_icon="💊",
    layout="wide",
)

st.title("💊 MedAgent-RAG")
st.caption("의약품 정보 · 약물 상호작용 · 복용 안전성 질문에 답변합니다")

# ── 세션 상태 초기화 ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ── 사이드바: Agent 흐름 ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Agent 실행 흐름")

    if st.session_state.messages:
        last = st.session_state.messages[-1]
        if last["role"] == "assistant":
            meta = last.get("meta", {})

            query_type = meta.get("query_type", "")
            if query_type:
                label_map = {
                    "simple": ("🔵", "단순 정보 조회"),
                    "interaction": ("🟠", "약물 상호작용"),
                    "safety": ("🔴", "복용 안전성"),
                    "complex": ("🟣", "복합 질의"),
                }
                icon, label = label_map.get(query_type, ("⚪", query_type))
                st.markdown(f"**질의 유형** {icon} `{label}`")

            trace = meta.get("agent_trace", [])
            if trace:
                st.markdown("**실행 순서**")
                agent_icons = {
                    "query_rewrite": "📝 Query Rewrite",
                    "supervisor": "🎯 Supervisor",
                    "drug_search": "🔍 Drug Search",
                    "grader": "📋 Grader",
                    "crag_rewrite": "🔄 CRAG Rewrite",
                    "interaction": "⚡ Interaction",
                    "safety": "🛡️ Safety",
                    "answer": "✍️ Answer",
                }
                for i, agent in enumerate(trace):
                    label = agent_icons.get(agent, agent)
                    connector = "└─" if i == len(trace) - 1 else "├─"
                    st.markdown(f"`{connector}` {label}")

            citations = meta.get("citations", [])
            if citations:
                st.markdown("---")
                st.markdown("**참조 출처**")
                for c in citations:
                    idx = c.get("index", "")
                    name = c.get("item_name", "")
                    source = c.get("source", "")
                    st.markdown(f"**[{idx}]** {name}")
                    st.caption(f"  {source}")
    else:
        st.info("질문을 입력하면 Agent 실행 흐름이 여기에 표시됩니다")

    st.markdown("---")
    st.caption(f"세션 ID: `{st.session_state.thread_id[:8]}...`")
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ── 예시 질문 버튼 ────────────────────────────────────────────────────────────
examples = [
    "타이레놀 효능이 뭐야?",
    "타이레놀이랑 아스피린 같이 먹어도 돼?",
    "임산부가 먹을 수 있는 감기약 뭐가 있어?",
    "혈압약 먹고 있는데 두통약 추천해줘",
]

if not st.session_state.messages:
    st.markdown("**예시 질문**")
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_query = ex
            st.rerun()

# ── 대화 히스토리 출력 ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            citations = msg.get("meta", {}).get("citations", [])
            if citations:
                with st.expander(f"참조 출처 ({len(citations)}건)"):
                    for c in citations:
                        idx = c.get("index", "")
                        name = c.get("item_name", "")
                        source = c.get("source", "")
                        preview = c.get("preview", "")
                        st.markdown(f"**[{idx}] {name}**  \n`{source}`")
                        if preview:
                            st.caption(preview)

# ── SSE 스트리밍 호출 ─────────────────────────────────────────────────────────


def call_api_stream(query: str, thread_id: str):
    """FastAPI SSE 엔드포인트를 호출하여 스트리밍 응답을 받음."""
    url = f"{API_BASE_URL}/v1/query/stream"
    payload = {"query": query, "thread_id": thread_id}

    answer_parts = []
    meta = {}

    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]  # "data: " 이후
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("event", "")

                if event_type == "node_start":
                    yield event.get("data", "")
                elif event_type == "answer":
                    text = event.get("data", "")
                    answer_parts.append(text)
                    yield text
                elif event_type == "done":
                    meta["query_type"] = event.get("query_type", "")
                    meta["agent_trace"] = event.get("agent_trace", [])
                    meta["citations"] = event.get("citations", [])
                    meta["thread_id"] = event.get("thread_id", thread_id)
                elif event_type == "error":
                    yield f"\n오류: {event.get('data', '')}"

    # 메타 데이터를 세션에 임시 저장
    st.session_state._last_meta = meta
    st.session_state._last_answer = "".join(answer_parts)


# ── 입력 처리 ─────────────────────────────────────────────────────────────────
query = st.chat_input("의약품에 대해 질문하세요")

# 예시 버튼으로 들어온 질의 처리
if "pending_query" in st.session_state and st.session_state.pending_query:
    query = st.session_state.pop("pending_query")

if query:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Agent 실행 (SSE 스트리밍)
    with st.chat_message("assistant"):
        try:
            st.write_stream(call_api_stream(query, thread_id=st.session_state.thread_id))
            meta = getattr(st.session_state, "_last_meta", {})
            answer = getattr(st.session_state, "_last_answer", "")
        except httpx.ConnectError:
            answer = "API 서버에 연결할 수 없습니다. FastAPI 서버가 실행 중인지 확인하세요."
            meta = {}
            st.markdown(answer)
        except Exception as e:
            answer = f"오류가 발생했습니다: {e}"
            meta = {}
            st.markdown(answer)

    # 메시지 저장
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": {
            "query_type": meta.get("query_type", ""),
            "agent_trace": meta.get("agent_trace", []),
            "citations": meta.get("citations", []),
        },
    })

    # 임시 데이터 정리
    st.session_state.pop("_last_meta", None)
    st.session_state.pop("_last_answer", None)

    st.rerun()
