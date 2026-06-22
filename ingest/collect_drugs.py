"""공공데이터포털 '의약품 e약은요' API 수집기.

원본 MedAgent-RAG의 동일 모듈을 포팅하되, 결과를 로컬 파일이 아니라 GCS에
직접 업로드한다. 후속 인덱싱(Vertex AI Search, Vector Search)이 GCS를
공통 소스로 참조하기 위함.

Usage:
    python -m ingest.collect_drugs
"""

import json
import time
from urllib.parse import unquote

import requests
from google.cloud import storage
from tqdm import tqdm

from config.settings import settings

BASE_URL = "http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
NUM_OF_ROWS = 100
RAW_GCS_KEY = "drugs/raw/drugs_raw.json"


def _api_key() -> str:
    """공공데이터포털은 키를 URL-encoded 형태로 발급하는데, requests의 params는
    자동으로 한 번 더 인코딩하므로 우선 디코딩해서 raw 상태로 만든다."""
    if not settings.DATA_API_KEY:
        raise RuntimeError("DATA_API_KEY 환경변수가 설정되지 않았습니다.")
    return unquote(settings.DATA_API_KEY)


def fetch_page(page_no: int) -> dict:
    params = {
        "serviceKey": _api_key(),
        "type": "json",
        "numOfRows": NUM_OF_ROWS,
        "pageNo": page_no,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def collect_all() -> list[dict]:
    first_page = fetch_page(1)
    body = first_page["body"]
    total_count = body["totalCount"]
    total_pages = (total_count + NUM_OF_ROWS - 1) // NUM_OF_ROWS

    print(f"전체 {total_count}건, {total_pages}페이지 수집 시작")

    all_items: list[dict] = list(body["items"])
    for page in tqdm(range(2, total_pages + 1), desc="수집"):
        time.sleep(0.3)  # rate limit 여유
        data = fetch_page(page)
        items = data["body"].get("items") or []
        all_items.extend(items)

    print(f"수집 완료: {len(all_items)}건")
    return all_items


def upload_to_gcs(items: list[dict], gcs_key: str = RAW_GCS_KEY) -> str:
    """원본 JSON을 데이터 버킷에 업로드. gs:// URI 반환."""
    bucket_name = settings.GCS_DATA_BUCKET.removeprefix("gs://").rstrip("/")
    client = storage.Client(project=settings.GCP_PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_key)
    blob.upload_from_string(
        json.dumps(items, ensure_ascii=False),
        content_type="application/json; charset=utf-8",
    )
    uri = f"gs://{bucket_name}/{gcs_key}"
    print(f"업로드 완료: {uri} ({len(items)}건)")
    return uri


def main() -> None:
    items = collect_all()
    upload_to_gcs(items)


if __name__ == "__main__":
    main()
