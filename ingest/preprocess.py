"""원본 e약은요 JSON → Vertex AI Search / Vector Search 양쪽 모두 쓰는 JSONL.

각 라인은 다음 스키마:
{
    "id": "<itemSeq>",
    "structData": {
        "item_name": "...",
        "company": "...",
        "update_date": "...",
        "categories": ["효능효과", "용법용량", ...]
    },
    "content": "<concatenated long-form text for embedding/search>"
}

- `structData`는 Vertex AI Search 필터/패싯용
- `content`는 임베딩(text-multilingual-embedding-002) + Search 본문

Usage:
    python -m ingest.preprocess
"""

import json
import re

from google.cloud import storage

from config.settings import settings

RAW_GCS_KEY = "drugs/raw/drugs_raw.json"
PROCESSED_GCS_KEY = "drugs/processed/drugs.jsonl"


_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean(text: str | None) -> str:
    if not text:
        return ""
    text = _HTML_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


_FIELDS = [
    ("efcyQesitm", "효능효과"),
    ("useMethodQesitm", "용법용량"),
    ("atpnWarnQesitm", "경고"),
    ("atpnQesitm", "주의사항"),
    ("intrcQesitm", "상호작용"),
    ("seQesitm", "부작용"),
    ("depositMethodQesitm", "보관방법"),
]


def process_item(item: dict) -> dict | None:
    item_name = (item.get("itemName") or "").strip()
    if not item_name:
        return None

    sections: list[tuple[str, str]] = []
    for key, label in _FIELDS:
        value = _clean(item.get(key))
        if value:
            sections.append((label, value))

    if not sections:
        return None  # 본문 없는 항목 스킵

    company = (item.get("entpName") or "").strip()
    item_seq = str(item.get("itemSeq", "")).strip()
    update_date = (item.get("updateDe") or "").strip()

    content_lines = [f"제품명: {item_name}"]
    if company:
        content_lines.append(f"제조사: {company}")
    content_lines.extend(f"{label}: {value}" for label, value in sections)
    content = "\n".join(content_lines)

    return {
        "id": item_seq or item_name,  # itemSeq 없으면 fallback
        "structData": {
            "item_name": item_name,
            "company": company,
            "update_date": update_date,
            "categories": [label for label, _ in sections],
        },
        "content": content,
    }


def _gcs_client() -> storage.Client:
    return storage.Client(project=settings.GCP_PROJECT_ID)


def _bucket():
    name = settings.GCS_DATA_BUCKET.removeprefix("gs://").rstrip("/")
    return _gcs_client().bucket(name)


def load_raw(gcs_key: str = RAW_GCS_KEY) -> list[dict]:
    blob = _bucket().blob(gcs_key)
    raw = blob.download_as_text(encoding="utf-8")
    return json.loads(raw)


def save_jsonl(records: list[dict], gcs_key: str = PROCESSED_GCS_KEY) -> str:
    blob = _bucket().blob(gcs_key)
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    blob.upload_from_string(
        payload, content_type="application/jsonl; charset=utf-8"
    )
    uri = f"gs://{_bucket().name}/{gcs_key}"
    print(f"업로드 완료: {uri} ({len(records)} 라인)")
    return uri


def main() -> None:
    raw_items = load_raw()
    print(f"원본 {len(raw_items)}건 로드")

    processed: list[dict] = []
    seen_ids: set[str] = set()
    for item in raw_items:
        rec = process_item(item)
        if rec is None:
            continue
        if rec["id"] in seen_ids:
            continue
        seen_ids.add(rec["id"])
        processed.append(rec)

    avg_len = sum(len(r["content"]) for r in processed) / max(len(processed), 1)
    print(f"전처리 {len(processed)}건, 평균 본문 {avg_len:.0f}자")
    save_jsonl(processed)


if __name__ == "__main__":
    main()
