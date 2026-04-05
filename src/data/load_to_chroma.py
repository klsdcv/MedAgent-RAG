"""전처리된 의약품 데이터를 ChromaDB에 적재.

Triton 서버가 실행 중이어야 합니다:
    docker compose -f docker/docker-compose.triton.yml up -d

Usage:
    python -m src.data.load_to_chroma
"""

import json
from pathlib import Path

import chromadb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.vectorstore.triton_embedder import TritonEmbedder
from src.config.settings import CHROMA_DB_PATH, CHROMA_COLLECTION_DRUGS

PROCESSED_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "drugs_processed.json"
BATCH_SIZE = 20  # Triton에 한 번에 보내는 건수


def load_to_chroma():
    with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"적재 대상: {len(items)}건")

    # ChromaDB 클라이언트
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # 기존 컬렉션 있으면 삭제 후 재생성
    try:
        client.delete_collection(CHROMA_COLLECTION_DRUGS)
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION_DRUGS,
        metadata={"hnsw:space": "cosine"},
    )

    # Triton 임베딩
    embedder = TritonEmbedder()

    # 배치 단위로 적재
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i : i + BATCH_SIZE]

        ids = [item["id"] for item in batch]
        documents = [item["document"] for item in batch]
        metadatas = [item["metadata"] for item in batch]

        embeddings = embedder.embed_documents(documents)

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        loaded = min(i + BATCH_SIZE, len(items))
        print(f"  [{loaded}/{len(items)}] 적재 완료")

    print(f"\nChromaDB 적재 완료: {collection.count()}건")
    print(f"저장 경로: {CHROMA_DB_PATH}")

    # 검색 테스트
    print("\n--- 검색 테스트 ---")
    query = "두통약 추천"
    query_embedding = embedder.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )
    for idx, (doc_id, doc, dist) in enumerate(
        zip(results["ids"][0], results["documents"][0], results["distances"][0])
    ):
        name = doc.split("\n")[0]
        print(f"  {idx+1}. [{dist:.4f}] {name}")


if __name__ == "__main__":
    load_to_chroma()
