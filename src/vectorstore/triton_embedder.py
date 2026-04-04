"""Triton Inference Server를 통한 BGE-M3 임베딩 클라이언트."""

import numpy as np
import requests
from transformers import AutoTokenizer


class TritonEmbedder:
    """Triton HTTP API로 BGE-M3 임베딩을 생성하는 클라이언트."""

    def __init__(
        self,
        triton_url: str = "http://localhost:8000",
        model_name: str = "bge_m3",
        tokenizer_name: str = "BAAI/bge-m3",
        max_length: int = 512,
    ):
        self.triton_url = triton_url.rstrip("/")
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _tokenize(self, texts: list[str]) -> dict:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        return {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
            "token_type_ids": encoded.get(
                "token_type_ids",
                np.zeros_like(encoded["input_ids"]),
            ).astype(np.int64),
        }

    def _build_request(self, tokens: dict) -> dict:
        inputs = []
        for name, arr in tokens.items():
            inputs.append({
                "name": name,
                "shape": list(arr.shape),
                "datatype": "INT64",
                "data": arr.tolist(),
            })

        return {
            "inputs": inputs,
            "outputs": [{"name": "last_hidden_state"}],
        }

    def _mean_pooling(
        self, hidden_states: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        mask_expanded = np.expand_dims(attention_mask, axis=-1)
        sum_embeddings = np.sum(hidden_states * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, a_min=1e-9, a_max=None)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트 리스트를 임베딩 벡터로 변환."""
        tokens = self._tokenize(texts)
        payload = self._build_request(tokens)

        url = f"{self.triton_url}/v2/models/{self.model_name}/infer"
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        output_data = result["outputs"][0]["data"]
        output_shape = result["outputs"][0]["shape"]

        hidden_states = np.array(output_data).reshape(output_shape)
        pooled = self._mean_pooling(hidden_states, tokens["attention_mask"])
        normalized = self._normalize(pooled)

        return normalized.tolist()

    def embed_query(self, text: str) -> list[float]:
        """단일 쿼리 임베딩."""
        return self.embed([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """문서 리스트 임베딩 (ChromaDB EmbeddingFunction 인터페이스 호환)."""
        results = []
        for text in texts:
            results.append(self.embed([text])[0])
        return results
