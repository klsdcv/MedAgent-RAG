"""단순 RAG(baseline) 평가 스크립트.

전체 멀티에이전트 시스템과의 비교를 위한 베이스라인.
- 검색: ChromaDB 벡터 검색 top-k 단일 (하이브리드/리랭킹/CRAG/멀티에이전트 없음)
- 답변: 전체 시스템과 동일한 모델(gpt-4o)·동일 ANSWER_SYSTEM_PROMPT
- 평가: 동일 평가셋 + 동일 RAGAS judge(gpt-4o)

Usage:
    python scripts/run_baseline_eval.py --save
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config.settings import OPENAI_API_KEY, OPENAI_MODEL
from src.config.prompts import ANSWER_SYSTEM_PROMPT
from src.agents.drug_search import search_vector
from src.evaluation import evaluator as ev

BASELINE_RESULTS_PATH = Path(__file__).parent.parent / "data" / "eval" / "baseline_results.json"
FULL_RESULTS_PATH = Path(__file__).parent.parent / "data" / "eval" / "eval_results.json"
TOP_K = 5

_answer_llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.3)


def naive_rag(question: str) -> tuple[str, list[str]]:
    """단순 RAG: 벡터검색 top-k → 컨텍스트 stuffing → 단일 LLM 답변."""
    hits = search_vector(question, n_results=TOP_K)
    contexts = [h.get("document", "") for h in hits]

    numbered = "\n\n".join(f"[{i}] {doc}" for i, doc in enumerate(contexts, 1))
    user_msg = f"참고 정보:\n{numbered}\n\n질문: {question}"
    resp = _answer_llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])
    return resp.content, contexts


def collect_baseline(eval_items: list[dict]) -> list[dict]:
    records = []
    total = len(eval_items)
    for i, item in enumerate(eval_items, 1):
        q = item["question"]
        print(f"  [{i}/{total}] {q[:40]}...", flush=True)
        try:
            answer, contexts = naive_rag(q)
            if not contexts:
                contexts = ["관련 정보 없음"]
        except Exception as e:
            print(f"    오류: {e}", flush=True)
            answer, contexts = "", ["오류로 인해 검색 실패"]
        records.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
            "query_type": item.get("query_type", ""),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="단순 RAG 베이스라인 평가")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    with open(ev.EVAL_PATH, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    print(f"평가 데이터: {len(eval_items)}건 (단순 RAG 베이스라인)")
    print("단순 RAG 실행 중...\n")
    records = collect_baseline(eval_items)

    print("\nRAGAS 평가 실행 중 (judge=gpt-4o)...")
    ragas_result = ev.run_ragas(records)

    df = ragas_result.to_pandas()
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def _nanmean(values):
        nums = [float(v) for v in values if v is not None and not (v != v)]
        return sum(nums) / len(nums) if nums else 0.0

    scores = {m: round(_nanmean(df[m].tolist()), 4) for m in metric_names if m in df.columns}

    output = {"scores": scores, "records": records}
    if args.save:
        BASELINE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n베이스라인 결과 저장: {BASELINE_RESULTS_PATH}")

    # 전체 시스템과 비교
    print("\n" + "=" * 60)
    print("  단순 RAG (baseline)  vs  멀티에이전트 시스템")
    print("=" * 60)
    full = None
    if FULL_RESULTS_PATH.exists():
        full = json.load(open(FULL_RESULTS_PATH, encoding="utf-8"))["scores"]

    labels = {
        "faithfulness": "Faithfulness     ",
        "answer_relevancy": "Answer Relevancy ",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall   ",
    }
    print(f"\n  {'지표':<18} {'baseline':>10} {'system':>10} {'개선':>12}")
    for m in metric_names:
        b = scores.get(m)
        s = full.get(m) if full else None
        if b is None or s is None:
            continue
        if b > 0:
            delta_pct = (s - b) / b * 100
            delta_str = f"{delta_pct:+.1f}%"
        else:
            delta_str = "n/a"
        print(f"  {labels[m]:<18} {b:>10.4f} {s:>10.4f} {delta_str:>12}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
