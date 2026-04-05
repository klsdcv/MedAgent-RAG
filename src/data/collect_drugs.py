"""e약은요 API에서 의약품 데이터를 수집하는 스크립트.

Usage:
    python -m src.data.collect_drugs
"""

import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("DATA_API_KEY")
BASE_URL = "http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
NUM_OF_ROWS = 100  # 한 페이지당 건수


def fetch_page(page_no: int) -> dict:
    """API에서 한 페이지 데이터를 가져온다."""
    params = {
        "serviceKey": API_KEY,
        "type": "json",
        "numOfRows": NUM_OF_ROWS,
        "pageNo": page_no,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def collect_all() -> list[dict]:
    """전체 데이터를 페이지 단위로 수집한다."""
    first_page = fetch_page(1)
    total_count = first_page["body"]["totalCount"]
    total_pages = (total_count // NUM_OF_ROWS) + 1
    print(f"전체 {total_count}건, {total_pages}페이지 수집 시작")

    all_items = first_page["body"]["items"]
    print(f"  [1/{total_pages}] {len(all_items)}건")

    for page in range(2, total_pages + 1):
        time.sleep(0.5)  # rate limit
        data = fetch_page(page)
        items = data["body"]["items"]
        all_items.extend(items)
        print(f"  [{page}/{total_pages}] 누적 {len(all_items)}건")

    return all_items


def save(items: list[dict], filename: str = "drugs_raw.json"):
    """수집 데이터를 JSON 파일로 저장한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {output_path} ({len(items)}건)")


if __name__ == "__main__":
    items = collect_all()
    save(items)
