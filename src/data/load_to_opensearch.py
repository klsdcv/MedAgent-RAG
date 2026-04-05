"""전처리된 의약품 데이터를 OpenSearch에 적재.

Usage:
    python -m src.data.load_to_opensearch
"""

import json
from pathlib import Path

from opensearchpy.helpers import bulk

from src.vectorstore.opensearch_client import get_client, create_index, INDEX_NAME

PROCESSED_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "drugs_processed.json"


def load_to_opensearch():
    with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"적재 대상: {len(items)}건")

    client = get_client()
    create_index(client)

    # bulk 적재
    actions = []
    for item in items:
        action = {
            "_index": INDEX_NAME,
            "_id": item["id"],
            "_source": {
                "item_name": item["metadata"]["item_name"],
                "company": item["metadata"]["company"],
                "item_seq": item["metadata"]["item_seq"],
                "update_date": item["metadata"].get("update_date", ""),
                "document": item["document"],
            },
        }
        actions.append(action)

    success, errors = bulk(client, actions, chunk_size=500, raise_on_error=False)
    print(f"적재 완료: {success}건 성공, {len(errors)}건 실패")

    # 검색 테스트
    client.indices.refresh(index=INDEX_NAME)
    count = client.count(index=INDEX_NAME)["count"]
    print(f"인덱스 문서 수: {count}건")

    print("\n--- 검색 테스트 ---")
    from src.vectorstore.opensearch_client import search
    results = search("타이레놀", n_results=3, client=client)
    for r in results:
        print(f"  {r['metadata']['item_name']} (score: {r['bm25_score']:.2f})")


if __name__ == "__main__":
    load_to_opensearch()
