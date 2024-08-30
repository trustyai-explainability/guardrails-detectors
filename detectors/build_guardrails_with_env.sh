#!/bin/bash

# Load variables from the .env file
set -o allexport
source .env
set +o allexport

# Define the path to the Dockerfile
DOCKERFILE_PATH="Dockerfile.guardrails-ai"

# Define the image name and tag
IMAGE_NAME="my-guardrails-app"
TAG="latest"

# Build the Docker image using the specified Dockerfile
podman build \
    --no-cache   -f $DOCKERFILE_PATH \
    --build-arg GUARDRAILS_METRICS=$GUARDRAILS_METRICS \
    --build-arg GUARDRAILS_REMOTE_INFERENCING=$GUARDRAILS_REMOTE_INFERENCING \
    --build-arg GUARDRAILS_API_KEY=$GUARDRAILS_API_KEY \
    -t ${IMAGE_NAME}:${TAG} .
