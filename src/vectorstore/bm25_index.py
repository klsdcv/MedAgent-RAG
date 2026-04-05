"""BM25 키워드 검색 인덱스."""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


PROCESSED_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "drugs_processed.json"


def tokenize(text: str) -> list[str]:
    """한국어/영어 텍스트를 토큰화 (부분 매칭을 위한 n-gram 포함)."""
    text = text.lower()
    # 특수문자를 공백으로
    text = re.sub(r"[^\w\s가-힣a-z0-9]", " ", text)
    tokens = [t for t in text.split() if len(t) > 1]

    # 한글 토큰에서 2~4글자 서브워드 추가 (약물명 부분매칭용)
    subwords = []
    for t in tokens:
        if re.search(r"[가-힣]", t) and len(t) > 3:
            for n in range(2, min(len(t), 5)):
                for i in range(len(t) - n + 1):
                    sub = t[i:i + n]
                    if len(sub) >= 2:
                        subwords.append(sub)

    return tokens + subwords


class BM25Index:
    """의약품 데이터용 BM25 검색 인덱스."""

    def __init__(self, data_path: str | Path | None = None):
        self._documents: list[dict] = []
        self._bm25: BM25Okapi | None = None

        path = Path(data_path) if data_path else PROCESSED_PATH
        if path.exists():
            self._load(path)

    def _load(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            self._documents = json.load(f)

        corpus = [tokenize(doc["document"]) for doc in self._documents]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """BM25로 키워드 검색."""
        if not self._bm25:
            return []

        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # 상위 n개 인덱스
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self._documents[idx]
                results.append({
                    "id": doc["id"],
                    "document": doc["document"],
                    "metadata": doc["metadata"],
                    "bm25_score": float(scores[idx]),
                })

        return results
