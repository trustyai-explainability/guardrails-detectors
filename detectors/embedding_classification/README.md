# Embedding Classification Detector

# Setup
1) Fetch prerequisite models, train pipeline, save training artifacts
   ```bash
   cd guardrails-detectors/detectors/embedding_classification/build
   make all
   ```
2) Build image (this can take a while and use a lot of VM storage during the build, beware):
    ```bash
    cd guardrails-detectors
    podman build --file=Dockerfile.embedding-classifier -t mmlu_detector:latest
    ```
   
## Testing Locally

```bash
podman run -p 8001:8000 --platform=linux/amd64 quay.io/rgeada/mmlu_detector:latest
```
wait for the server to start (you should see a log message like `Uvicorn running on http://0.0.0.0:8000`), then:
```bash
curl -X POST "localhost:8001/api/v1/text/contents" -H "Content-Type: application/json" \
-H "detector-id: mmluTopicMatch" \
-d '{
    "contents": ["How far away is the Sun from the center of the Milky Way?", "What is the healthiest vegetable?", "The square root of 256 is 16."],
         "allowList": ["astronomy"],
         "blockList": ["nutrition"]
}' | jq
```