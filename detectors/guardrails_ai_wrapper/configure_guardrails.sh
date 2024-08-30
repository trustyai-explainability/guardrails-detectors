#!/bin/bash

# Load environment variables from .env file
set -o allexport
source /app/.env
set +o allexport

# Check if required environment variables are set
if [ -z "$GUARDRAILS_METRICS" ] || [ -z "$GUARDRAILS_REMOTE_INFERENCING" ] || [ -z "$GUARDRAILS_API_KEY" ]; then
    echo "Required environment variables are missing."
    exit 1
fi

# Run guardrails configure with the environment variables
guardrails configure <<EOF
$GUARDRAILS_METRICS
$GUARDRAILS_REMOTE_INFERENCING
$GUARDRAILS_API_KEY
EOF
