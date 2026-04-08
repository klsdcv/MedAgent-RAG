"""BGE Cross-Encoder 기반 Reranker.

hybrid search (Vector + BM25 + RRF) 결과를 cross-encoder로 재랭킹.
Triton Inference Server가 가용하면 GPU 추론, 아니면 CPU fallback.
"""

import os

import numpy as np
import requests
from transformers import AutoTokenizer


_TRITON_URL = os.getenv("TRITON_URL", "http://localhost:8000")
_RERANKER_MODEL = "bge_reranker"
_TOKENIZER_ID = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """BAAI/bge-reranker-v2-m3 기반 cross-encoder reranker.

    Triton 서버가 연결 가능하면 GPU 추론(ONNX),
    연결 불가 시 sentence-transformers CPU fallback.
    """

    MODEL_NAME = _TOKENIZER_ID

    def __init__(self) -> None:
        self._tokenizer = None
        self._cpu_model = None
        self._use_triton: bool | None = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID)
        return self._tokenizer

    def _check_triton(self) -> bool:
        """Triton 서버의 reranker 모델 가용 여부를 확인."""
        try:
            resp = requests.get(
                f"{_TRITON_URL}/v2/models/{_RERANKER_MODEL}/ready",
                timeout=2,
            )
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def _is_triton_available(self) -> bool:
        if self._use_triton is None:
            self._use_triton = self._check_triton()
        return self._use_triton

    def _triton_predict(self, query: str, docs: list[dict]) -> list[float]:
        """Triton HTTP API로 reranker 점수를 계산."""
        tokenizer = self._get_tokenizer()
        pairs = [[query, doc["document"]] for doc in docs]
        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        payload = {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": list(input_ids.shape),
                    "datatype": "INT64",
                    "data": input_ids.tolist(),
                },
                {
                    "name": "attention_mask",
                    "shape": list(attention_mask.shape),
                    "datatype": "INT64",
                    "data": attention_mask.tolist(),
                },
            ],
        }

        resp = requests.post(
            f"{_TRITON_URL}/v2/models/{_RERANKER_MODEL}/infer",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

        logits = result["outputs"][0]["data"]
        # logits shape: [batch_size, 1] — 각 pair의 relevance score
        # 1차원 리스트로 반환
        if isinstance(logits[0], list):
            return [row[0] for row in logits]
        return logits

    def _cpu_predict(self, query: str, docs: list[dict]) -> list[float]:
        """CPU sentence-transformers CrossEncoder fallback."""
        if self._cpu_model is None:
            from sentence_transformers import CrossEncoder
            self._cpu_model = CrossEncoder(self.MODEL_NAME)

        pairs = [(query, doc["document"]) for doc in docs]
        scores = self._cpu_model.predict(pairs)
        return [float(s) for s in scores]

    def rerank(self, query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
        """hybrid search 결과를 cross-encoder 점수로 재랭킹.

        Args:
            query: 사용자 질의
            docs: hybrid_search()가 반환한 문서 리스트
            top_k: 반환할 상위 문서 수

        Returns:
            rerank_score 필드가 추가된 상위 top_k 문서 리스트
        """
        if not docs:
            return docs

        if self._is_triton_available():
            try:
                scores = self._triton_predict(query, docs)
            except Exception:
                # Triton 실패 시 CPU fallback
                self._use_triton = False
                scores = self._cpu_predict(query, docs)
        else:
            scores = self._cpu_predict(query, docs)

        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
