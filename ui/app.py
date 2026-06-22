"""MedAgent RAG Streamlit UI.

배포된 Vertex AI Agent Engine을 호출하여 사용자 질의에 대한 멀티에이전트
응답을 표시한다. Cloud Run 배포 대상.

로컬 실행:
    streamlit run ui/app.py

Cloud Run 배포: deploy/05_deploy_ui_cloudrun.sh 참조.
"""

from __future__ import annotations

import os
import time

import streamlit as st
import vertexai
from vertexai import agent_engines

# Cloud Run 환경에서는 ADC를 통해 자동 인증되므로 명시적 키 불필요
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


def render() -> None:
    st.set_page_config(
        page_title="MedAgent RAG (Vertex AI)",
        page_icon="💊",
        layout="centered",
    )
    st.title("💊 MedAgent RAG")
    st.caption("한국 의약품 상담 멀티에이전트 — Google ADK + Vertex AI")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.user_id = f"streamlit-{int(time.time())}"
        st.session_state.session_id = None

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

    remote_app = get_remote_app()
    if st.session_state.session_id is None:
        sess = remote_app.create_session(user_id=st.session_state.user_id)
        st.session_state.session_id = sess["id"]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        chunks: list[str] = []
        with st.spinner("멀티에이전트 응답 생성 중…"):
            for event in remote_app.stream_query(
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                message=prompt,
            ):
                content = event.get("content") or {}
                for part in content.get("parts", []):
                    text = part.get("text")
                    if text:
                        chunks.append(text)
                        placeholder.markdown("".join(chunks))

        final = "".join(chunks)
        if not final:
            final = "(응답이 비어 있습니다.)"
            placeholder.markdown(final)

    st.session_state.messages.append({"role": "assistant", "content": final})


if __name__ == "__main__":
    render()
