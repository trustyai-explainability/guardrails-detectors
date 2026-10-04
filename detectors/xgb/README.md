# XGB Classification Detector

## Setup
1. Train XGB model and save trained model
    ```
    cd guardrails-detectors/detectors/xgb/build
    make all
    ```

2. Build image
    ```
    cd guardrails-detectors
    podman build --file=Dockerfile.xgb -t xgb_detector:latest
    ```

## Detector API
## `/api/v1/text/contents`
*

## Testing Locally
```
podman run -p 8001:8000 --platform=linux/amd64 quay.io/christinaexyou/xgb_detector:latest
```

Wait for the server to start
```
```