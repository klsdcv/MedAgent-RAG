"""평가셋 20문항을 배포된 Agent Engine에 던져 답변·context를 수집.

원본 `data/eval/eval_dataset.json`을 그대로 사용 (simple/interaction/safety 3종).

각 문항당:
- Agent Engine `stream_query` 호출
- `function_response` 이벤트에서 검색 도구의 반환 dict 추출 → context 리스트로 누적
- 모델 text 합쳐 final answer

결과를 `eval/results/rag_outputs.json`에 저장. evaluate.py가 이를 읽어 RAGAS 메트릭 계산.

Usage:
    python -m eval.run_rag
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import vertexai
from vertexai import agent_engines

from config.settings import settings

EVAL_DATASET = Path("_reference/MedAgent-RAG/data/eval/eval_dataset.json")
RESULTS_DIR = Path("eval/results")
OUT_PATH = RESULTS_DIR / "rag_outputs.json"


def _extract_contexts(function_response_part: dict) -> list[str]:
    """search_drugs / hybrid_search_drugs 도구 응답에서 content 텍스트만 추출."""
    fr = function_response_part.get("function_response") or {}
    resp = fr.get("response") or {}
    payload = resp.get("result") if "result" in resp else resp
    out: list[str] = []
    if isinstance(payload, list):
        for r in payload:
            if isinstance(r, dict):
                text = r.get("content") or r.get("snippet")
                if text:
                    out.append(text)
    return out


def run_one(app, question: str, user_id: str) -> tuple[str, list[str]]:
    sess = app.create_session(user_id=user_id)
    answer_parts: list[str] = []
    contexts: list[str] = []
    seen_context_ids: set[int] = set()

    for event in app.stream_query(
        user_id=user_id,
        session_id=sess["id"],
        message=question,
    ):
        content = event.get("content") or {}
        for part in content.get("parts", []):
            if part.get("text"):
                # supervisor의 최종 답변과 sub-agent 답변 모두 합침
                answer_parts.append(part["text"])
            elif part.get("function_response"):
                for ctx in _extract_contexts(part):
                    key = hash(ctx)
                    if key not in seen_context_ids:
                        seen_context_ids.add(key)
                        contexts.append(ctx)
    return "".join(answer_parts).strip(), contexts


def main() -> None:
    vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
    app = agent_engines.get(settings.AGENT_ENGINE_RESOURCE_NAME)

    dataset = json.loads(EVAL_DATASET.read_text(encoding="utf-8"))
    print(f"평가 문항: {len(dataset)}개")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[dict] = []

    for i, item in enumerate(dataset, start=1):
        q = item["question"]
        qtype = item["query_type"]
        gt = item["ground_truth"]
        uid = f"eval-{int(time.time())}-{i:02d}"

        print(f"\n[{i:02d}/{len(dataset)}] ({qtype}) {q}")
        try:
            answer, contexts = run_one(app, q, uid)
            print(f"  ↳ contexts={len(contexts)}, answer_len={len(answer)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ↳ 실패: {exc!r}")
            answer, contexts = "", []

        outputs.append({
            "query_type": qtype,
            "question": q,
            "ground_truth": gt,
            "answer": answer,
            "contexts": contexts,
        })

        # 중간 저장 (장애 대비)
        OUT_PATH.write_text(
            json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"\n완료. 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
