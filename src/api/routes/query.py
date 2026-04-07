"""Query 관련 API 라우트."""

import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import QueryRequest, QueryResponse, SessionResponse
from src.graph.workflow import run_query, stream_query, _app

router = APIRouter(prefix="/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """일반 질의 엔드포인트."""
    thread_id = req.thread_id or str(uuid.uuid4())

    try:
        result = await asyncio.to_thread(run_query, req.query, thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"워크플로 실행 오류: {e}")

    citations = []
    for c in result.get("citations", []):
        citations.append({
            "index": c.get("index", 0),
            "item_name": c.get("item_name", ""),
            "source": c.get("source", ""),
            "preview": c.get("preview", ""),
        })

    return QueryResponse(
        query=req.query,
        rewritten_query=result.get("rewritten_query", ""),
        query_type=result.get("query_type", ""),
        final_answer=result.get("final_answer", ""),
        citations=citations,
        agent_trace=result.get("agent_trace", []),
        thread_id=thread_id,
    )


@router.post("/query/stream")
async def query_stream_endpoint(req: QueryRequest):
    """SSE 스트리밍 엔드포인트."""
    thread_id = req.thread_id or str(uuid.uuid4())

    # stream_query는 동기 generator — 별도 스레드에서 실행하고 큐로 전달
    queue: asyncio.Queue = asyncio.Queue()

    def _run_sync():
        try:
            for chunk in stream_query(req.query, thread_id=thread_id):
                queue.put_nowait(chunk)
            queue.put_nowait(None)  # sentinel: 완료
        except Exception as e:
            queue.put_nowait(e)

    async def event_generator():
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, _run_sync)

        try:
            while True:
                item = await queue.get()

                if item is None:
                    break
                if isinstance(item, Exception):
                    error_data = json.dumps({"event": "error", "data": str(item)}, ensure_ascii=False)
                    yield f"data: {error_data}\n\n"
                    return

                stripped = item.strip()
                if stripped.startswith("`") and stripped.endswith("`"):
                    event_type = "node_start"
                else:
                    event_type = "answer"

                data = json.dumps({"event": event_type, "data": item}, ensure_ascii=False)
                yield f"data: {data}\n\n"

            await task  # 스레드 완료 대기

            # 완료 이벤트 — 최종 상태 포함
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = _app.get_state(config)
            result = dict(snapshot.values) if snapshot and snapshot.values else {}

            done_data = json.dumps({
                "event": "done",
                "data": "",
                "query_type": result.get("query_type", ""),
                "agent_trace": result.get("agent_trace", []),
                "citations": result.get("citations", []),
                "thread_id": thread_id,
            }, ensure_ascii=False)
            yield f"data: {done_data}\n\n"

        except Exception as e:
            error_data = json.dumps({"event": "error", "data": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/session/{thread_id}", response_model=SessionResponse)
async def session_endpoint(thread_id: str):
    """세션 상태 조회."""
    config = {"configurable": {"thread_id": thread_id}}

    try:
        snapshot = _app.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

    state = dict(snapshot.values)
    return SessionResponse(
        thread_id=thread_id,
        query_type=state.get("query_type", ""),
        agent_trace=state.get("agent_trace", []),
        messages=state.get("messages", []),
    )
