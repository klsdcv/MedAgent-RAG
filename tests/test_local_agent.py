"""ADK 루트 에이전트 로컬 invoke 테스트. Agent Engine 배포 전 검증용.

Usage:
    python -m tests.test_local_agent "활명수 효능 알려줘"
"""

from __future__ import annotations

import asyncio
import os
import sys

from config.settings import settings

# ADK 내부 google-genai가 Vertex 모드로 동작하도록 환경변수 설정
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = settings.GCP_PROJECT_ID
os.environ["GOOGLE_CLOUD_LOCATION"] = settings.GCP_LOCATION

import vertexai  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from agents.root_agent import root_agent  # noqa: E402


async def run(query: str) -> None:
    vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)

    runner = InMemoryRunner(agent=root_agent, app_name="med-rag-local")
    session = await runner.session_service.create_session(
        app_name="med-rag-local",
        user_id="local-test",
    )

    msg = types.Content(role="user", parts=[types.Part.from_text(text=query)])
    print(f"질의: {query}\n")
    print("=" * 60)

    async for event in runner.run_async(
        user_id="local-test",
        session_id=session.id,
        new_message=msg,
    ):
        author = event.author or "?"
        content = event.content
        if not content:
            continue
        for part in content.parts or []:
            if getattr(part, "text", None):
                print(f"[{author}] {part.text}", flush=True)
            elif getattr(part, "function_call", None):
                fc = part.function_call
                print(f"[{author}] → tool: {fc.name}({dict(fc.args)})")
            elif getattr(part, "function_response", None):
                fr = part.function_response
                print(f"[{author}] ← tool {fr.name} resp ({len(str(fr.response))} bytes)")


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "활명수 효능 알려줘"
    asyncio.run(run(query))


if __name__ == "__main__":
    main()
