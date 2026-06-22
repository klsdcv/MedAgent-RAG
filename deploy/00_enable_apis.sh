#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="civic-athlete-500200-t0"
REGION="us-central1"

echo ">>> Setting project: ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}"
gcloud config set ai/region "${REGION}"
gcloud config set run/region "${REGION}"

echo ">>> Enabling APIs (takes 1-2 min)"
gcloud services enable \
  aiplatform.googleapis.com \
  discoveryengine.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  compute.googleapis.com

echo ">>> Verifying ADC"
gcloud auth application-default print-access-token > /dev/null && echo "ADC OK"

echo ">>> Default service account"
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')
echo "Project number: ${PROJECT_NUMBER}"
echo "Compute SA: ${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo ">>> Done. Enabled APIs:"
gcloud services list --enabled --filter="name:(aiplatform OR discoveryengine OR run OR secretmanager OR storage OR artifactregistry)" --format="table(config.name)"
