"""중간 JSONL → Vertex AI Search(Discovery Engine) 데이터스토어로 임포트.

흐름:
1) `gs://.../drugs/processed/drugs.jsonl` (id/structData/content) 로드
2) Discovery Engine import 포맷으로 변환 후 `drugs/search/import.jsonl`에 저장
   - structured 데이터스토어용: `{id, jsonData: "<JSON string>"}` 한 줄
3) GENERIC 데이터스토어가 없으면 생성 (location=global, content_config=NO_CONTENT)
4) ImportDocumentsRequest로 GCS에서 import (long-running)

Discovery Engine은 한국어 본문도 structData 안에 그대로 두고 검색에 활용 가능.
1차 구현은 structured 방식으로 단순화한다.

Usage:
    python -m ingest.load_to_search
"""

import json
import time

from google.api_core.exceptions import AlreadyExists
from google.cloud import discoveryengine_v1 as de
from google.cloud import storage

from config.settings import settings

PROCESSED_GCS_KEY = "drugs/processed/drugs.jsonl"
IMPORT_GCS_KEY = "drugs/search/import.jsonl"
LOCATION = "global"  # Discovery Engine 데이터스토어는 global 권장


def _bucket_name() -> str:
    return settings.GCS_DATA_BUCKET.removeprefix("gs://").rstrip("/")


def build_import_jsonl() -> str:
    """중간 JSONL → Discovery Engine structured 포맷으로 변환 후 GCS 업로드."""
    client = storage.Client(project=settings.GCP_PROJECT_ID)
    bucket = client.bucket(_bucket_name())

    src_blob = bucket.blob(PROCESSED_GCS_KEY)
    raw_text = src_blob.download_as_text(encoding="utf-8")

    lines: list[str] = []
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        # data_schema="custom"이면 JSONL 각 라인이 _id를 포함한 임의 JSON
        out = {
            "_id": rec["id"],
            **rec["structData"],
            "content": rec["content"],
        }
        lines.append(json.dumps(out, ensure_ascii=False))

    payload = "\n".join(lines)
    dst_blob = bucket.blob(IMPORT_GCS_KEY)
    dst_blob.upload_from_string(
        payload, content_type="application/jsonl; charset=utf-8"
    )
    uri = f"gs://{_bucket_name()}/{IMPORT_GCS_KEY}"
    print(f"import JSONL 업로드: {uri} ({len(lines)} 라인)")
    return uri


def ensure_data_store() -> str:
    """데이터스토어가 없으면 생성. data_store 리소스 경로 반환."""
    client = de.DataStoreServiceClient()
    parent = client.collection_path(
        project=settings.GCP_PROJECT_ID,
        location=LOCATION,
        collection=settings.SEARCH_COLLECTION,
    )

    data_store_id = settings.SEARCH_DATA_STORE_ID
    data_store = de.DataStore(
        display_name="MedAgent Drugs",
        industry_vertical=de.IndustryVertical.GENERIC,
        solution_types=[de.SolutionType.SOLUTION_TYPE_SEARCH],
        content_config=de.DataStore.ContentConfig.NO_CONTENT,
    )

    try:
        op = client.create_data_store(
            parent=parent,
            data_store=data_store,
            data_store_id=data_store_id,
        )
        print(f"데이터스토어 생성 중: {data_store_id}")
        ds = op.result(timeout=300)
        print(f"생성 완료: {ds.name}")
        return ds.name
    except AlreadyExists:
        name = client.data_store_path(
            project=settings.GCP_PROJECT_ID,
            location=LOCATION,
            data_store=data_store_id,
        )
        # collection 포함 경로로 정정
        name = (
            f"projects/{settings.GCP_PROJECT_ID}"
            f"/locations/{LOCATION}"
            f"/collections/{settings.SEARCH_COLLECTION}"
            f"/dataStores/{data_store_id}"
        )
        print(f"이미 존재: {name}")
        return name


def import_documents(data_store_name: str, gcs_uri: str) -> None:
    """GCS JSONL에서 데이터스토어로 import (long-running)."""
    client = de.DocumentServiceClient()
    parent = f"{data_store_name}/branches/default_branch"

    request = de.ImportDocumentsRequest(
        parent=parent,
        gcs_source=de.GcsSource(
            input_uris=[gcs_uri],
            data_schema="custom",  # jsonData 기반 structured
        ),
        reconciliation_mode=de.ImportDocumentsRequest.ReconciliationMode.FULL,
    )

    print(f"문서 import 시작: {gcs_uri}")
    op = client.import_documents(request=request)

    # long-running, 폴링
    start = time.time()
    while not op.done():
        elapsed = int(time.time() - start)
        print(f"  ... {elapsed}s 경과")
        time.sleep(15)
    result = op.result()
    print(f"import 완료: errors={result.error_samples or '없음'}")


def main() -> None:
    gcs_uri = build_import_jsonl()
    data_store_name = ensure_data_store()
    import_documents(data_store_name, gcs_uri)


if __name__ == "__main__":
    main()
