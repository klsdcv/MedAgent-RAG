"""MedAgent-RAG 평가 실행 스크립트.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --save          # 결과를 JSON으로 저장
    python scripts/run_eval.py --type simple   # 특정 유형만 평가
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import run_evaluation, print_report

RESULTS_PATH = Path(__file__).parent.parent / "data" / "eval" / "eval_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="MedAgent-RAG 평가 실행")
    parser.add_argument("--save", action="store_true", help="결과를 JSON으로 저장")
    parser.add_argument(
        "--type",
        choices=["simple", "interaction", "safety"],
        help="특정 질의 유형만 평가",
    )
    args = parser.parse_args()

    # 특정 유형 필터링
    if args.type:
        original_path = Path(__file__).parent.parent / "data" / "eval" / "eval_dataset.json"
        with open(original_path, "r", encoding="utf-8") as f:
            all_items = json.load(f)
        filtered = [item for item in all_items if item["query_type"] == args.type]
        print(f"필터링: {args.type} 유형 {len(filtered)}건")

        # 임시 파일 생성
        tmp_path = Path(__file__).parent.parent / "data" / "eval" / "_tmp_eval.json"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False)

        # evaluator의 EVAL_PATH 임시 교체
        import src.evaluation.evaluator as ev
        original_eval_path = ev.EVAL_PATH
        ev.EVAL_PATH = tmp_path

    save_path = RESULTS_PATH if args.save else None
    output = run_evaluation(save_path=save_path)
    print_report(output)

    if args.type:
        ev.EVAL_PATH = original_eval_path
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
