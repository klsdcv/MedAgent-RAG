"""Drug Search Agent - 의약품 정보 벡터 검색 (RAG)."""

import chromadb
from langchain_openai import ChatOpenAI

from src.config.settings import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_DRUGS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from src.config.prompts import DRUG_SEARCH_SYSTEM_PROMPT
from src.graph.state import MedAgentState
from src.vectorstore.triton_embedder import TritonEmbedder


embedder = TritonEmbedder()


def search_drugs(query: str, n_results: int = 5) -> list[dict]:
    """ChromaDB에서 의약품 정보를 검색."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(CHROMA_COLLECTION_DRUGS)

    query_embedding = embedder.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    items = []
    for doc_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        items.append({
            "id": doc_id,
            "document": doc,
            "metadata": meta,
            "similarity": 1 - dist,  # cosine distance → similarity
        })

    return items


def drug_search_node(state: MedAgentState) -> dict:
    """Drug Search Agent 노드 함수."""
    query = state["query"]
    results = search_drugs(query)

    return {
        "drug_results": results,
        "agent_trace": state.get("agent_trace", []) + ["drug_search"],
    }
