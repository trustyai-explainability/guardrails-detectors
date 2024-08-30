#!/bin/bash

# Install packages from Guardrails Hub
guardrails hub install hub://guardrails/regex_match
if [ $? -ne 0 ]; then
    echo "Failed to install regex_match"
    exit 1
fi

guardrails hub install hub://guardrails/competitor_check
if [ $? -ne 0 ]; then
    echo "Failed to install competitor_check"
    exit 1
fi

guardrails hub install hub://guardrails/toxic_language
if [ $? -ne 0 ]; then
    echo "Failed to install toxic_language"
    exit 1
fi
