curl -ks -X POST localhost:8000/api/v1/text/contents \
   -H "Content-Type: application/json" \
    -H "detector-id: huggingface_model" \
    -d '{
        "contents": ["You are too kind my dear."]
    }'