# Streamlit UI for MedAgent RAG (Cloud Run)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Vertex SDK + Streamlit만 설치 (ADK 런타임은 Agent Engine에 있으므로 불필요)
RUN pip install --no-cache-dir \
    "streamlit>=1.38.0" \
    "google-cloud-aiplatform[agent_engines]>=1.71.0" \
    "pydantic-settings>=2.5.0"

COPY ui/app.py /app/app.py

ENV PORT=8080 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

CMD ["streamlit", "run", "/app/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
