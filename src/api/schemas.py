"""API 요청/응답 Pydantic 모델."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="사용자 질의", min_length=1, max_length=500)
    thread_id: str | None = Field(None, description="대화 세션 ID (없으면 새 세션 생성)")


class Citation(BaseModel):
    index: int
    item_name: str = ""
    source: str = ""
    preview: str = ""


class QueryResponse(BaseModel):
    query: str = Field(..., description="원본 질의")
    rewritten_query: str = Field("", description="전처리된 질의")
    query_type: str = Field("", description="질의 유형")
    final_answer: str = Field("", description="최종 답변")
    citations: list[Citation] = Field(default_factory=list)
    agent_trace: list[str] = Field(default_factory=list)
    thread_id: str = Field("", description="세션 ID")


class SessionResponse(BaseModel):
    thread_id: str
    query_type: str = ""
    agent_trace: list[str] = Field(default_factory=list)
    messages: list[dict] = Field(default_factory=list)


class StreamEvent(BaseModel):
    """SSE 이벤트 데이터."""
    event: str = Field(..., description="이벤트 타입: node_start | answer | done | error")
    data: str = Field("", description="이벤트 데이터")
