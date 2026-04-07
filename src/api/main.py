"""MedAgent-RAG FastAPI 백엔드."""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.query import router as query_router

app = FastAPI(
    title="MedAgent-RAG API",
    description="의약품 정보 Multi-Agent RAG 시스템 API",
    version="1.0.0",
)

# CORS 설정 (Streamlit 등 프론트엔드 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
