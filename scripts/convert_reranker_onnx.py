"""BGE-Reranker-v2-M3 모델을 ONNX로 변환하는 스크립트.

Usage:
    python scripts/convert_reranker_onnx.py

출력:
    triton_models/bge_reranker/1/model.onnx
"""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def convert():
    model_id = "BAAI/bge-reranker-v2-m3"
    output_dir = Path(__file__).parent.parent / "triton_models" / "bge_reranker" / "1"
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    print(f"BGE-Reranker-v2-M3 모델 다운로드 중: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()

    # 더미 입력 생성 (query-document pair)
    # batch=1 더미는 ONNX export 시 batch 차원이 상수(1)로 고정되는 문제가 있어
    # batch>=2 더미로 트레이싱하여 dynamic batch를 보존한다.
    dummy_pairs = [
        ["약물 상호작용", "아세트아미노펜과 이부프로펜의 병용"],
        ["복용 안전성", "타이레놀은 공복에 복용해도 되나요"],
    ]
    inputs = tokenizer(
        dummy_pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"ONNX 변환 시작 → {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            opset_version=17,
        )

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"변환 완료: {onnx_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    convert()
