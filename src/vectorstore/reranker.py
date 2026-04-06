"""BGE Cross-Encoder 기반 Reranker.

hybrid search (Vector + BM25 + RRF) 결과를 cross-encoder로 재랭킹.
"""


class Reranker:
    """BAAI/bge-reranker-v2-m3 기반 cross-encoder reranker.

    한국어·영어 모두 지원하며 의약품 문서 재랭킹에 적합.
    모델은 첫 호출 시 lazy 로딩.
    """

    MODEL_NAME = "BAAI/bge-reranker-v2-m3"

    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.MODEL_NAME)
        return self._model

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

        model = self._get_model()
        pairs = [(query, doc["document"]) for doc in docs]
        scores = model.predict(pairs)

        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
