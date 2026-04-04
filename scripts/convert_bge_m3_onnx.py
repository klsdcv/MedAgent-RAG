"""BGE-M3 모델을 ONNX로 변환하는 스크립트.

Usage:
    python scripts/convert_bge_m3_onnx.py

출력:
    triton_models/bge_m3/1/model.onnx
"""

from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def convert():
    model_id = "BAAI/bge-m3"
    output_dir = Path(__file__).parent.parent / "triton_models" / "bge_m3" / "1"
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    print(f"BGE-M3 모델 다운로드 중: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    # 더미 입력 생성
    dummy_text = "안녕하세요"
    inputs = tokenizer(dummy_text, return_tensors="pt")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs.get(
        "token_type_ids", torch.zeros_like(input_ids)
    )

    print(f"ONNX 변환 시작 → {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask, token_type_ids),
            str(onnx_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "token_type_ids": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=17,
        )

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"변환 완료: {onnx_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    convert()
