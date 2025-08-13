## Hugging Face Detector Examples

### Detecting toxic content using Hugging Face Detectors

1. Set model variables and download the model locally, for example to store the [HAP Detector](https://huggingface.co/ibm-granite/granite-guardian-hap-38m) in a `hf-detectors` directory:

```bash
export HF_MODEL=ibm-granite/granite-guardian-hap-38m
export DETECTOR_STORAGE=hf-detectors
export DETECTOR_NAME=$(basename "$HF_MODEL")
export DETECTOR_DIR=$DETECTOR_STORAGE/$DETECTOR_NAME

huggingface-cli download "$HF_MODEL" --local-dir "$DETECTOR_DIR"
```

the instructions above assume you have [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli) installed, which you can do inside your Python virtual environment:

```bash
pip install "huggingface_hub[cli]"
```

2. Build the image for the Hugging Face Detector:

```bash
export HF_IMAGE=hf-detector:latest
podman build -f detectors/Dockerfile.hf -t $HF_IMAGE detectors
```

3. Run the detector container, mounting the model directory you downloaded in the previous step:

```bash
podman run --rm -p 8000:8000 \
  -e MODEL_DIR=/mnt/models/$DETECTOR_NAME \
  -v $(pwd)/$DETECTOR_DIR:/mnt/models/$DETECTOR_NAME:Z \
  $HF_IMAGE
```

4. Invoke the detector with a POST request; in a separate terminal, run:

```bash
curl -X POST \
  http://localhost:8000/api/v1/text/contents \
  -H 'accept: application/json' \
  -H 'detector-id: hap' \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": ["You dotard, I really hate this stuff", "I simply love this stuff"],
    "detector_params": {}
  }' | jq
```

5. You should see a response like this:

```json
[
  [
    {
      "start": 0,
      "end": 36,
      "detection": "sequence_classifier",
      "detection_type": "sequence_classification",
      "score": 0.9634233713150024,
      "sequence_classification": "LABEL_1",
      "sequence_probability": 0.9634233713150024,
      "token_classifications": null,
      "token_probabilities": null,
      "text": "You dotard, I really hate this stuff",
      "evidences": []
    }
  ],
  [
    {
      "start": 0,
      "end": 24,
      "detection": "sequence_classifier",
      "detection_type": "sequence_classification",
      "score": 0.00016677979147061706,
      "sequence_classification": "LABEL_0",
      "sequence_probability": 0.00016677979147061706,
      "token_classifications": null,
      "token_probabilities": null,
      "text": "I simply love this stuff",
      "evidences": []
    }
  ]
]
```

### Detecting prompt injection content using Hugging Face Detectors

- Following the steps above, you can readily use the Hugging Face Detector with a different model, such as the [prompt injection classifier](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2)

```bash
export HF_MODEL=protectai/deberta-v3-base-prompt-injection-v2
export DETECTOR_STORAGE=hf-detectors
export DETECTOR_NAME=$(basename "$HF_MODEL")
export DETECTOR_DIR=$DETECTOR_STORAGE/$DETECTOR_NAME

huggingface-cli download "$HF_MODEL" --local-dir "$DETECTOR_DIR"
```

- Build the image again to include the new model:

```bash
export HF_IMAGE=hf-detector:latest
podman build -f detectors/Dockerfile.hf -t $HF_IMAGE detectors
```

- then spin up the container as before:

```bash
podman run --rm -p 8000:8000 \
  -e MODEL_DIR=/mnt/models/$DETECTOR_NAME \
  -v $(pwd)/$DETECTOR_DIR:/mnt/models/$DETECTOR_NAME:Z \
  $HF_IMAGE
```

- and invoke the detector with a POST request; in a separate terminal, run:

```bash
curl -X POST \
  http://localhost:8000/api/v1/text/contents \
  -H 'accept: application/json' \
  -H 'detector-id: prompt-injection' \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": ["Trolol?", "How to make a delicious espresso?"],
    "detector_params": {}
  }' | jq
```

which should yield a response like this:

```json
[
  [
    {
      "start": 0,
      "end": 48,
      "detection": "sequence_classifier",
      "detection_type": "sequence_classification",
      "score": 0.9998816251754761,
      "sequence_classification": "INJECTION",
      "sequence_probability": 0.9998816251754761,
      "token_classifications": null,
      "token_probabilities": null,
      "text": "Trolol?",
      "evidences": []
    }
  ],
  [
    {
      "start": 0,
      "end": 33,
      "detection": "sequence_classifier",
      "detection_type": "sequence_classification",
      "score": 9.671030056779273E-7,
      "sequence_classification": "SAFE",
      "sequence_probability": 9.671030056779273E-7,
      "token_classifications": null,
      "token_probabilities": null,
      "text": "How to make a delicious espresso?",
      "evidences": []
    }
  ]
]
```
