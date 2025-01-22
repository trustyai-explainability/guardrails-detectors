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
   
## Detector API
### `/api/v1/text/contents`
* `contents`: List of texts to classify
* `allowList`: Allowed list of subjects: all inbound texts must belong to _at least one_ of these subjects to avoid flagging the detector
* `blockList`: Blocked list of subjects: all inbounds texts must not belong to _any_ of these subjects to avoid flagging the detector.
* `threshold`: Defines the maximum distance a body of text can be from the subject centroid and still be classified into that subject. The default value is 0.75, while a threshold of >10 will classify every document into every subject. As such, values 0<threshold<1 are recommended. 


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

