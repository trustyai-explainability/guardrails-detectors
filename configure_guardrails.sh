#!/bin/bash

# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)

# Set additional environment variables for guardrails configuration
export GUARDRAILS_METRICS=$GUARDRAILS_METRICS
export GUARDRAILS_REMOTE_INFERENCING=$GUARDRAILS_REMOTE_INFERENCING
export GUARDRAILS_API_KEY=$GUARDRAILS_API_KEY

# Run guardrails configure with the environment variables
guardrails configure <<EOF
$GUARDRAILS_METRICS
$GUARDRAILS_REMOTE_INFERENCING
$GUARDRAILS_API_KEY
EOF