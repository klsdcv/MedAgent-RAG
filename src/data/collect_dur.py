"""DUR(의약품안전사용) 데이터를 수집하는 스크립트.

수집 대상:
- 병용금기
- 특정연령대금기
- 임부금기

Usage:
    python -m src.data.collect_dur
"""

import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("DATA_API_KEY")
BASE_URL = "https://apis.data.go.kr/1471000/DURPrdlstInfoService03"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
NUM_OF_ROWS = 100

# 수집할 DUR 유형별 엔드포인트
DUR_TYPES = {
    "병용금기": "getUsjntTabooInfoList03",
    "특정연령대금기": "getSpcifyAgrdeTabooInfoList03",
    "임부금기": "getPwnmTabooInfoList03",
}


def fetch_page(endpoint: str, page_no: int) -> dict:
    url = f"{BASE_URL}/{endpoint}"
    params = {
        "serviceKey": API_KEY,
        "type": "json",
        "numOfRows": NUM_OF_ROWS,
        "pageNo": page_no,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def collect_type(type_name: str, endpoint: str, max_pages: int = 0) -> list[dict]:
    """특정 DUR 유형의 전체 데이터를 수집."""
    first_page = fetch_page(endpoint, 1)
    total_count = first_page["body"]["totalCount"]
    total_pages = (total_count // NUM_OF_ROWS) + 1

    if max_pages > 0:
        total_pages = min(total_pages, max_pages)

    print(f"[{type_name}] 전체 {total_count}건, {total_pages}페이지 수집")

    all_items = first_page["body"]["items"]
    print(f"  [1/{total_pages}] {len(all_items)}건")

    for page in range(2, total_pages + 1):
        time.sleep(0.5)
        data = fetch_page(endpoint, page)
        items = data["body"]["items"]
        all_items.extend(items)

        if page % 10 == 0 or page == total_pages:
            print(f"  [{page}/{total_pages}] 누적 {len(all_items)}건")

    return all_items


def save(items: list[dict], filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"저장: {output_path} ({len(items)}건)")


if __name__ == "__main__":
    for type_name, endpoint in DUR_TYPES.items():
        items = collect_type(type_name, endpoint)
        filename = f"dur_{endpoint.replace('get', '').replace('InfoList03', '').lower()}.json"
        save(items, filename)
        print()
