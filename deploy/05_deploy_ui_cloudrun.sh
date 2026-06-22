#!/usr/bin/env bash
set -euo pipefail

# Streamlit UI를 Cloud Run에 배포.
# 사전 조건:
#   - Agent Engine 배포 완료 (.env의 AGENT_ENGINE_RESOURCE_NAME 채워짐)
#   - bash deploy/00_enable_apis.sh 완료

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# .env 로드
set -a
source .env
set +a

PROJECT_ID="${GCP_PROJECT_ID}"
REGION="${GCP_LOCATION:-us-central1}"
SERVICE="med-rag-ui"
SA="${PROJECT_NUMBER:-719526263781}-compute@developer.gserviceaccount.com"

if [[ -z "${AGENT_ENGINE_RESOURCE_NAME:-}" ]]; then
    echo "ERROR: AGENT_ENGINE_RESOURCE_NAME이 .env에 비어있습니다."
    exit 1
fi

echo ">>> Cloud Run 배포 시작 (Buildpacks 또는 Dockerfile)"
gcloud run deploy "${SERVICE}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --source=. \
    --service-account="${SA}" \
    --allow-unauthenticated \
    --port=8080 \
    --memory=1Gi \
    --cpu=1 \
    --max-instances=3 \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_LOCATION=${REGION},AGENT_ENGINE_RESOURCE_NAME=${AGENT_ENGINE_RESOURCE_NAME}"

echo
echo "=========================================="
echo "배포 완료. 서비스 URL:"
gcloud run services describe "${SERVICE}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --format="value(status.url)"
echo "=========================================="
