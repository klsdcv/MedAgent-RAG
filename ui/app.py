"""MedAgent RAG Streamlit UI.

배포된 Vertex AI Agent Engine을 호출하여 멀티에이전트 응답을 표시한다.
Cloud Run 배포 대상.

세션 영속화:
- `user_id`는 URL `?uid=...` query param으로 안정화 (탭/새로고침 가로질러 유지)
- Agent Engine의 영속 SessionService를 사용 — `list_sessions(user_id=...)`로
  사용자별 최근 세션을 재사용. 사이드바의 "새 대화" 버튼으로 새 session 생성.
- 세션 자체는 Vertex AI 인프라에서 관리되므로 컨테이너 재시작에도 보존됨.
"""

from __future__ import annotations

import os
import uuid

import streamlit as st
import vertexai
from vertexai import agent_engines

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
AGENT_RESOURCE_NAME = os.environ.get("AGENT_ENGINE_RESOURCE_NAME", "")


@st.cache_resource
def get_remote_app():
    if not PROJECT_ID or not AGENT_RESOURCE_NAME:
        st.error("GCP_PROJECT_ID와 AGENT_ENGINE_RESOURCE_NAME 환경변수가 필요합니다.")
        st.stop()
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    return agent_engines.get(AGENT_RESOURCE_NAME)


def _ensure_user_id() -> str:
    """URL query param `uid`에서 user_id를 읽거나 새로 생성하여 set."""
    qp = st.query_params
    uid = qp.get("uid")
    if not uid:
        uid = f"u-{uuid.uuid4().hex[:12]}"
        st.query_params["uid"] = uid
    return uid


def _resolve_session(remote_app, user_id: str, force_new: bool = False) -> str:
    """기존 세션 있으면 가장 최근 것 reuse, 없거나 force_new면 새로 생성."""
    if not force_new:
        try:
            sessions = remote_app.list_sessions(user_id=user_id)
            session_list = sessions.get("sessions", []) if isinstance(sessions, dict) else list(sessions)
            if session_list:
                # 최근 갱신 우선
                latest = max(session_list, key=lambda s: s.get("last_update_time", 0))
                return latest["id"]
        except Exception as exc:  # noqa: BLE001
            st.toast(f"list_sessions 실패, 새 세션 생성: {exc!r}", icon="⚠️")
    sess = remote_app.create_session(user_id=user_id)
    return sess["id"]


def _load_session_history(remote_app, user_id: str, session_id: str) -> list[dict]:
    """기존 세션 이벤트에서 user/assistant 메시지를 복원."""
    try:
        sess = remote_app.get_session(user_id=user_id, session_id=session_id)
    except Exception:
        return []
    events = sess.get("events", []) if isinstance(sess, dict) else []
    messages: list[dict] = []
    for ev in events:
        content = ev.get("content") or {}
        role_raw = content.get("role")
        text_pieces = [p.get("text", "") for p in content.get("parts", []) if p.get("text")]
        text = "".join(text_pieces).strip()
        if not text:
            continue
        role = "user" if role_raw == "user" else "assistant"
        messages.append({"role": role, "content": text})
    return messages


def render() -> None:
    st.set_page_config(
        page_title="MedAgent RAG (Vertex AI)",
        page_icon="💊",
        layout="centered",
    )
    st.title("💊 MedAgent RAG")
    st.caption("한국 의약품 상담 멀티에이전트 — Google ADK + Vertex AI · 하이브리드 RRF 검색")

    remote_app = get_remote_app()
    user_id = _ensure_user_id()

    # 사이드바 — 세션 제어
    with st.sidebar:
        st.markdown(f"**user_id**: `{user_id}`")
        st.caption("URL의 `?uid=...`로 안정화됨")
        if st.button("🆕 새 대화", use_container_width=True):
            new_id = _resolve_session(remote_app, user_id, force_new=True)
            st.session_state.session_id = new_id
            st.session_state.messages = []
            st.rerun()

    # 세션 결정 + 이력 복원
    if "session_id" not in st.session_state:
        st.session_state.session_id = _resolve_session(remote_app, user_id)
        st.session_state.messages = _load_session_history(
            remote_app, user_id, st.session_state.session_id
        )

    # 메시지 이력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("약품 정보·상호작용·복용 안전성에 대해 물어보세요")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        chunks: list[str] = []
        with st.spinner("멀티에이전트 응답 생성 중…"):
            for event in remote_app.stream_query(
                user_id=user_id,
                session_id=st.session_state.session_id,
                message=prompt,
            ):
                content = event.get("content") or {}
                for part in content.get("parts", []):
                    text = part.get("text")
                    if text:
                        chunks.append(text)
                        placeholder.markdown("".join(chunks))

        final = "".join(chunks) or "(응답이 비어 있습니다.)"
        placeholder.markdown(final)

    st.session_state.messages.append({"role": "assistant", "content": final})


if __name__ == "__main__":
    render()
