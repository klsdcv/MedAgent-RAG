"""수집된 의약품 데이터를 전처리하여 ChromaDB 적재용으로 변환.

Usage:
    python -m src.data.preprocess_drugs
"""

import json
import re
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "drugs_raw.json"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "drugs_processed.json"


def clean_html(text: str | None) -> str:
    """HTML 태그 제거 및 텍스트 정리."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_item(item: dict) -> dict | None:
    """단일 의약품 항목을 전처리."""
    item_name = (item.get("itemName") or "").strip()
    if not item_name:
        return None

    efcy = clean_html(item.get("efcyQesitm"))
    use_method = clean_html(item.get("useMethodQesitm"))
    atpn_warn = clean_html(item.get("atpnWarnQesitm"))
    atpn = clean_html(item.get("atpnQesitm"))
    intrc = clean_html(item.get("intrcQesitm"))
    side_effect = clean_html(item.get("seQesitm"))
    deposit = clean_html(item.get("depositMethodQesitm"))

    # 임베딩용 텍스트: 핵심 정보를 하나로 합침
    doc_text = f"제품명: {item_name}"
    if efcy:
        doc_text += f"\n효능효과: {efcy}"
    if use_method:
        doc_text += f"\n용법용량: {use_method}"
    if atpn_warn:
        doc_text += f"\n경고: {atpn_warn}"
    if atpn:
        doc_text += f"\n주의사항: {atpn}"
    if intrc:
        doc_text += f"\n상호작용: {intrc}"
    if side_effect:
        doc_text += f"\n부작용: {side_effect}"
    if deposit:
        doc_text += f"\n보관방법: {deposit}"

    return {
        "id": str(item.get("itemSeq", "")),
        "item_name": item_name,
        "company": (item.get("entpName") or "").strip(),
        "document": doc_text,
        "metadata": {
            "item_name": item_name,
            "company": (item.get("entpName") or "").strip(),
            "item_seq": str(item.get("itemSeq", "")),
            "update_date": item.get("updateDe", ""),
        },
    }


def preprocess():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    print(f"원본 데이터: {len(raw_items)}건")

    processed = []
    for item in raw_items:
        result = process_item(item)
        if result:
            processed.append(result)

    # 중복 제거 (itemSeq 기준)
    seen = set()
    unique = []
    for item in processed:
        if item["id"] not in seen:
            seen.add(item["id"])
            unique.append(item)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    print(f"전처리 완료: {len(unique)}건 → {OUTPUT_PATH}")

    # 통계
    avg_len = sum(len(d["document"]) for d in unique) / len(unique)
    print(f"평균 문서 길이: {avg_len:.0f}자")


if __name__ == "__main__":
    preprocess()
