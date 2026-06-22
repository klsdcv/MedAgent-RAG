#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-civic-athlete-500200-t0}"
REGION="${GCP_LOCATION:-us-central1}"

# Vertex Agent Engine 스테이징 + 원본 약품 JSONL 보관용 버킷
STAGING_BUCKET="${PROJECT_ID}-vertex-staging"
DATA_BUCKET="${PROJECT_ID}-med-data"

echo ">>> Creating buckets in ${REGION}"
for B in "${STAGING_BUCKET}" "${DATA_BUCKET}"; do
    if gcloud storage buckets describe "gs://${B}" >/dev/null 2>&1; then
        echo "exists: gs://${B}"
    else
        gcloud storage buckets create "gs://${B}" \
            --location="${REGION}" \
            --uniform-bucket-level-access
        echo "created: gs://${B}"
    fi
done

echo ">>> Done. Set in .env:"
echo "GCS_STAGING_BUCKET=gs://${STAGING_BUCKET}"
echo "GCS_DATA_BUCKET=gs://${DATA_BUCKET}"
