"""Redis 캐싱 클라이언트.

동일 질의 반복 시 LLM + 검색 재실행을 방지한다.
질의를 sha256 해시로 키를 생성하고 TTL 24시간으로 캐싱.
"""

import hashlib
import json
import os

import redis


_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_TTL_SECONDS = 60 * 60 * 24  # 24시간

_client: redis.Redis | None = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(_REDIS_URL, decode_responses=True)
    return _client


def _make_key(query: str) -> str:
    """질의 문자열을 sha256 해시 키로 변환."""
    h = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
    return f"medagent:query:{h}"


def get_cached_result(query: str) -> dict | None:
    """캐시된 질의 결과를 조회. 없으면 None 반환."""
    try:
        client = _get_client()
        data = client.get(_make_key(query))
        if data is not None:
            return json.loads(data)
    except redis.ConnectionError:
        return None
    return None


def set_cached_result(query: str, result: dict) -> None:
    """질의 결과를 캐시에 저장."""
    try:
        client = _get_client()
        # 직렬화 가능한 필드만 저장
        cacheable = {
            "query": result.get("query", ""),
            "original_query": result.get("original_query", ""),
            "query_type": result.get("query_type", ""),
            "drug_results": result.get("drug_results", []),
            "interaction_results": result.get("interaction_results", []),
            "safety_results": result.get("safety_results", []),
            "final_answer": result.get("final_answer", ""),
            "citations": result.get("citations", []),
            "agent_trace": result.get("agent_trace", []),
            "rewritten_query": result.get("rewritten_query", ""),
        }
        client.setex(_make_key(query), _TTL_SECONDS, json.dumps(cacheable, ensure_ascii=False))
    except redis.ConnectionError:
        pass
