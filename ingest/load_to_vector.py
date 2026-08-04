"""중간 JSONL → text-multilingual-embedding-002 임베딩 → Vector Search 입력 JSONL.

흐름:
1) `gs://.../drugs/processed/drugs.jsonl` 로드
2) 본문(content)을 배치 단위로 임베딩 (text-multilingual-embedding-002, 768d)
3) Vector Search 인덱스 입력 포맷으로 변환:
   {"id": "<itemSeq>", "embedding": [...], "restricts": [{"namespace": "company", "allow": ["..."]}]}
4) `gs://.../drugs/vector/embeddings.jsonl`에 업로드

본 모듈은 임베딩만 책임진다. 인덱스 생성/엔드포인트 배포는
`deploy/03_build_vector_index.py`에서 수행.

Usage:
    python -m ingest.load_to_vector
"""

import json
import time
from typing import Iterable

import vertexai
from google.cloud import storage
from tqdm import tqdm
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from config.settings import settings

PROCESSED_GCS_KEY = "drugs/processed/drugs.jsonl"
VECTOR_GCS_KEY = "drugs/vector/embeddings.jsonl"

MAX_CHARS_PER_DOC = 2000       # 평균 1000자, 긴 항목 truncate
MAX_CHARS_PER_BATCH = 22000    # 한국어 ≈ 0.72 token/char, 20k token 한도 안전 마진
MAX_DOCS_PER_BATCH = 20
EMBED_DIM = 768


def _bucket():
    name = settings.GCS_DATA_BUCKET.removeprefix("gs://").rstrip("/")
    return storage.Client(project=settings.GCP_PROJECT_ID).bucket(name)


def _truncate(text: str, max_chars: int = MAX_CHARS_PER_DOC) -> str:
    return text[:max_chars]


def _dynamic_batches(records: list[dict]) -> Iterable[list[dict]]:
    """글자수 누적 기준 동적 배치. 한국어 토큰 비효율에 대응."""
    batch: list[dict] = []
    total_chars = 0
    for rec in records:
        size = min(len(rec["content"]), MAX_CHARS_PER_DOC)
        over_chars = batch and total_chars + size > MAX_CHARS_PER_BATCH
        over_docs = len(batch) >= MAX_DOCS_PER_BATCH
        if over_chars or over_docs:
            yield batch
            batch = []
            total_chars = 0
        batch.append(rec)
        total_chars += size
    if batch:
        yield batch


def load_records() -> list[dict]:
    blob = _bucket().blob(PROCESSED_GCS_KEY)
    raw = blob.download_as_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def embed_batch(model: TextEmbeddingModel, texts: list[str]) -> list[list[float]]:
    """배치 임베딩. 일시적 에러만 지수 백오프 재시도 (InvalidArgument는 즉시 raise)."""
    from google.api_core.exceptions import InvalidArgument

    inputs = [
        TextEmbeddingInput(text=_truncate(t), task_type="RETRIEVAL_DOCUMENT")
        for t in texts
    ]
    delay = 2.0
    for attempt in range(3):
        try:
            resp = model.get_embeddings(inputs, output_dimensionality=EMBED_DIM)
            return [e.values for e in resp]
        except InvalidArgument:
            raise  # 토큰 초과 등은 재시도해도 똑같음
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                raise
            print(f"  embed 일시 실패({attempt + 1}/3): {exc!r} — {delay}s 후 재시도")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def main() -> None:
    vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
    model = TextEmbeddingModel.from_pretrained(settings.VERTEX_EMBED_MODEL)

    records = load_records()
    print(f"임베딩 대상: {len(records)}건")

    out_lines: list[str] = []
    batches = list(_dynamic_batches(records))
    print(f"동적 배치 {len(batches)}개 생성")
    for batch in tqdm(batches, desc="임베딩"):
        texts = [r["content"] for r in batch]
        vectors = embed_batch(model, texts)
        for rec, vec in zip(batch, vectors):
            sd = rec["structData"]
            restricts = []
            if sd.get("company"):
                restricts.append({"namespace": "company", "allow": [sd["company"]]})
            out = {
                "id": rec["id"],
                "embedding": vec,
                "restricts": restricts,
            }
            out_lines.append(json.dumps(out, ensure_ascii=False))

    payload = "\n".join(out_lines)
    blob = _bucket().blob(VECTOR_GCS_KEY)
    blob.upload_from_string(payload, content_type="application/jsonl; charset=utf-8")
    uri = f"gs://{_bucket().name}/{VECTOR_GCS_KEY}"
    print(f"업로드 완료: {uri} ({len(out_lines)} 벡터, 차원={EMBED_DIM})")


if __name__ == "__main__":
    main()
