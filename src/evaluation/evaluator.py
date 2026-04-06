"""RAGAS 기반 MedAgent-RAG 평가 파이프라인."""

import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.graph.workflow import run_query

EVAL_PATH = Path(__file__).parent.parent.parent / "data" / "eval" / "eval_dataset.json"


def collect_predictions(eval_items: list[dict]) -> list[dict]:
    """각 질문을 워크플로에 실행하여 답변과 컨텍스트를 수집."""
    records = []
    total = len(eval_items)

    for i, item in enumerate(eval_items, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"  [{i}/{total}] {question[:40]}...")

        try:
            result = run_query(question)
            answer = result.get("final_answer", "")

            # 검색된 문서를 contexts 리스트로 변환
            contexts = []
            for r in result.get("drug_results", [])[:3]:
                contexts.append(r.get("document", ""))
            for r in result.get("safety_results", [])[:2]:
                contexts.append(r.get("document", ""))

            if not contexts:
                contexts = ["관련 정보 없음"]

        except Exception as e:
            print(f"    오류 발생: {e}")
            answer = ""
            contexts = ["오류로 인해 검색 실패"]

        records.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "query_type": item.get("query_type", ""),
        })

    return records


def run_ragas(records: list[dict]) -> dict:
    """RAGAS 평가 실행."""
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in records],
        "answer": [r["answer"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result


def run_evaluation(save_path: Path | None = None) -> dict:
    """전체 평가 파이프라인 실행.

    Returns:
        {
            "scores": {metric: score, ...},
            "by_type": {query_type: {metric: score, ...}, ...},
            "records": [...],
        }
    """
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    print(f"평가 데이터: {len(eval_items)}건")
    print("워크플로 실행 중...\n")
    records = collect_predictions(eval_items)

    print("\nRAGAS 평가 실행 중...")
    ragas_result = run_ragas(records)

    # 전체 점수
    scores = {
        "faithfulness": float(ragas_result["faithfulness"]),
        "answer_relevancy": float(ragas_result["answer_relevancy"]),
        "context_precision": float(ragas_result["context_precision"]),
        "context_recall": float(ragas_result["context_recall"]),
    }

    # 유형별 점수
    by_type: dict[str, dict[str, list[float]]] = {}
    for i, record in enumerate(records):
        qt = record["query_type"]
        if qt not in by_type:
            by_type[qt] = {m: [] for m in scores}

        row = ragas_result.to_pandas().iloc[i]
        for metric in scores:
            val = row.get(metric)
            if val is not None and not (val != val):  # NaN 제외
                by_type[qt][metric].append(float(val))

    by_type_avg = {
        qt: {
            metric: round(sum(vals) / len(vals), 4) if vals else None
            for metric, vals in metrics.items()
        }
        for qt, metrics in by_type.items()
    }

    output = {
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "by_type": by_type_avg,
        "records": records,
    }

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {save_path}")

    return output


def print_report(output: dict) -> None:
    """평가 결과를 터미널에 출력."""
    scores = output["scores"]
    by_type = output["by_type"]

    print("\n" + "=" * 55)
    print("  MedAgent-RAG 평가 결과 (RAGAS)")
    print("=" * 55)

    metric_labels = {
        "faithfulness": "Faithfulness     (충실도)",
        "answer_relevancy": "Answer Relevancy (관련성)",
        "context_precision": "Context Precision(정밀도)",
        "context_recall": "Context Recall   (재현율)",
    }

    print("\n[전체]")
    for metric, label in metric_labels.items():
        score = scores.get(metric)
        bar = "█" * int((score or 0) * 20)
        print(f"  {label}: {score:.4f}  {bar}")

    print("\n[유형별]")
    type_labels = {"simple": "단순 정보", "interaction": "약물 상호작용", "safety": "복용 안전성"}
    for qt, metrics in by_type.items():
        label = type_labels.get(qt, qt)
        print(f"\n  [{label}]")
        for metric, score in metrics.items():
            if score is not None:
                print(f"    {metric_labels[metric]}: {score:.4f}")

    print("\n" + "=" * 55)
