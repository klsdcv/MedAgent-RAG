"""MedAgent-RAG Streamlit UI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from src.graph.workflow import run_query

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
                    "supervisor": "🎯 Supervisor",
                    "drug_search": "🔍 Drug Search",
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
                st.markdown("**참고 의약품**")
                for c in citations:
                    st.markdown(f"- {c['item_name']}")
    else:
        st.info("질문을 입력하면 Agent 실행 흐름이 여기에 표시됩니다")

    st.markdown("---")
    if st.button("대화 초기화"):
        st.session_state.messages = []
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

    # Agent 실행
    with st.chat_message("assistant"):
        with st.spinner("Agent 실행 중..."):
            try:
                result = run_query(query)
                answer = result.get("final_answer", "답변을 생성하지 못했습니다.")
            except Exception as e:
                answer = f"오류가 발생했습니다: {e}"
                result = {}

        st.markdown(answer)

    # 메시지 저장 (meta에 trace/citations 포함)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": {
            "query_type": result.get("query_type", ""),
            "agent_trace": result.get("agent_trace", []),
            "citations": result.get("citations", []),
        },
    })

    st.rerun()
