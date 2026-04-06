"""DUR 데이터 전처리 (특정연령대금기, 임부금기 → safety 컬렉션용).

Usage:
    python -m src.data.preprocess_dur
"""

import json
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "safety_processed.json"


def process_age_item(item: dict) -> dict:
    """특정연령대금기 항목 전처리."""
    item_name = (item.get("ITEM_NAME") or "").strip()
    ingr_name = (item.get("INGR_NAME") or "").strip()
    prohbt = (item.get("PROHBT_CONTENT") or "").strip()

    doc = f"제품명: {item_name}\n성분: {ingr_name}\n유형: 특정연령대금기\n금기내용: {prohbt}"

    return {
        "id": f"age_{item.get('ITEM_SEQ', '')}_{item.get('INGR_CODE', '')}",
        "document": doc,
        "metadata": {
            "type": "특정연령대금기",
            "item_name": item_name,
            "item_seq": str(item.get("ITEM_SEQ", "")),
            "ingr_name": ingr_name,
            "ingr_code": item.get("INGR_CODE", ""),
            "company": (item.get("ENTP_NAME") or "").strip(),
            "prohbt_content": prohbt,
        },
    }


def process_pregnancy_item(item: dict) -> dict:
    """임부금기 항목 전처리."""
    item_name = (item.get("ITEM_NAME") or "").strip()
    ingr_name = (item.get("INGR_NAME") or "").strip()
    prohbt = (item.get("PROHBT_CONTENT") or "").strip()

    doc = f"제품명: {item_name}\n성분: {ingr_name}\n유형: 임부금기\n금기내용: {prohbt}"

    return {
        "id": f"preg_{item.get('ITEM_SEQ', '')}_{item.get('INGR_CODE', '')}",
        "document": doc,
        "metadata": {
            "type": "임부금기",
            "item_name": item_name,
            "item_seq": str(item.get("ITEM_SEQ", "")),
            "ingr_name": ingr_name,
            "ingr_code": item.get("INGR_CODE", ""),
            "company": (item.get("ENTP_NAME") or "").strip(),
            "prohbt_content": prohbt,
        },
    }


def preprocess():
    all_items = []

    # 특정연령대금기
    age_path = RAW_DIR / "dur_spcifyagrdetaboo.json"
    if age_path.exists():
        with open(age_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            all_items.append(process_age_item(item))
        print(f"특정연령대금기: {len(raw)}건 전처리")

    # 임부금기
    preg_path = RAW_DIR / "dur_pwnmtaboo.json"
    if preg_path.exists():
        with open(preg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            all_items.append(process_pregnancy_item(item))
        print(f"임부금기: {len(raw)}건 전처리")

    # 중복 제거
    seen = set()
    unique = []
    for item in all_items:
        if item["id"] not in seen:
            seen.add(item["id"])
            unique.append(item)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    print(f"전체 {len(unique)}건 → {OUTPUT_PATH}")


if __name__ == "__main__":
    preprocess()
