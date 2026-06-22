"""RAGAS로 검색·답변 품질 정량화.

`eval/results/rag_outputs.json` (run_rag.py 출력)을 입력으로,
다음 4개 메트릭을 계산한다:

| 메트릭 | 측정 | 입력 |
| --- | --- | --- |
| faithfulness | 답변이 contexts에 충실한가 (할루시네이션 ↓) | answer, contexts |
| answer_relevancy | 답변이 질문에 관련 있는가 | question, answer |
| context_precision | retrieved context가 적절한가 (rank-aware) | question, contexts, ground_truth |
| context_recall | ground_truth 내용이 context에 다 있는가 | contexts, ground_truth |

평가용 LLM·임베딩은 Vertex Gemini Flash + `text-multilingual-embedding-002`.

Usage:
    python -m eval.evaluate
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# Shim: ragas 0.4.3은 langchain-community 0.x의 옛 경로
# `langchain_community.chat_models.vertexai`를 import한다. langchain-community 1.x
# 에서는 제거됐으므로, langchain-google-vertexai의 ChatVertexAI를 같은 경로에
# 노출하는 가짜 모듈을 sys.modules에 등록해 호환 유지.
import sys
import types

from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI

_shim = types.ModuleType("langchain_community.chat_models.vertexai")
_shim.ChatVertexAI = _ChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", _shim)
# ─────────────────────────────────────────────────────────────

import json
from pathlib import Path

from datasets import Dataset
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from config.settings import settings

INPUT_PATH = Path("eval/results/rag_outputs.json")
REPORT_JSON = Path("eval/results/ragas_report.json")
REPORT_MD = Path("eval/results/ragas_report.md")
SCORES_CSV = Path("eval/results/ragas_scores.csv")


def _build_dataset() -> Dataset:
    raw = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    # 빈 결과는 평가 불가 → context에 placeholder
    rows = []
    for r in raw:
        contexts = r["contexts"] if r["contexts"] else ["(검색 결과 없음)"]
        rows.append({
            "question": r["question"],
            "answer": r["answer"] or "(응답 없음)",
            "contexts": contexts,
            "ground_truth": r["ground_truth"],
            "query_type": r["query_type"],
        })
    return Dataset.from_list(rows)


def _llm_and_embeddings():
    llm = ChatVertexAI(
        model_name=settings.VERTEX_MODEL_GEMINI,
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
        temperature=0.0,
    )
    emb = VertexAIEmbeddings(
        model_name=settings.VERTEX_EMBED_MODEL,
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def _per_type_summary(scores_df) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    metric_cols = [c for c in scores_df.columns if c not in {"user_input", "retrieved_contexts", "response", "reference", "query_type", "question", "answer", "contexts", "ground_truth"}]
    for qtype, group in scores_df.groupby("query_type"):
        summary[qtype] = {m: round(float(group[m].mean()), 4) for m in metric_cols if m in group.columns}
    summary["_overall"] = {m: round(float(scores_df[m].mean()), 4) for m in metric_cols if m in scores_df.columns}
    return summary


def _render_markdown(summary: dict[str, dict[str, float]]) -> str:
    types_order = ["simple", "interaction", "safety", "_overall"]
    metrics_order = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    rows = []
    rows.append("# RAGAS 평가 보고서")
    rows.append("")
    rows.append("| query_type | " + " | ".join(metrics_order) + " |")
    rows.append("| --- | " + " | ".join("---" for _ in metrics_order) + " |")
    for t in types_order:
        if t not in summary:
            continue
        cells = [summary[t].get(m, float("nan")) for m in metrics_order]
        rows.append(f"| {t} | " + " | ".join(f"{c:.4f}" for c in cells) + " |")
    rows.append("")
    rows.append("- **faithfulness**: 답변이 검색 컨텍스트에 얼마나 충실한가 (↑ 좋음, 할루시네이션 ↓)")
    rows.append("- **answer_relevancy**: 답변이 질문에 얼마나 관련 있는가 (↑ 좋음)")
    rows.append("- **context_precision**: 검색 결과의 관련성/순서 적절성 (↑ 좋음, rank-aware)")
    rows.append("- **context_recall**: ground_truth 정보가 context에 얼마나 들어있는가 (↑ 좋음)")
    return "\n".join(rows)


def main() -> None:
    raw = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    ds = _build_dataset()
    print(f"평가 대상: {len(ds)} 문항")

    llm, emb = _llm_and_embeddings()

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    print("RAGAS 평가 실행 (LLM 호출 다수, 5~20분 소요)…")
    # RAGAS 기본 60s timeout이 Vertex Gemini 콜드부터에 짧음 → 300s로 완화 +
    # 동시성 줄여 토큰 한도/quota도 여유
    run_config = RunConfig(timeout=300, max_workers=4, max_retries=3)
    result = evaluate(
        ds,
        metrics=metrics,
        llm=llm,
        embeddings=emb,
        raise_exceptions=False,
        show_progress=True,
        run_config=run_config,
    )

    df = result.to_pandas()
    # RAGAS는 metadata 열을 떨어뜨리므로 직접 복원
    df["query_type"] = [r["query_type"] for r in raw][: len(df)]
    df["question"] = [r["question"] for r in raw][: len(df)]

    # 원본 점수 raw 저장 (실패 분석용)
    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORES_CSV, index=False)
    print(f"raw scores 저장: {SCORES_CSV}")

    summary = _per_type_summary(df)

    REPORT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(_render_markdown(summary), encoding="utf-8")

    # 콘솔 요약
    print()
    print(_render_markdown(summary))
    print()
    print(f"저장: {REPORT_JSON} / {REPORT_MD}")


if __name__ == "__main__":
    main()
