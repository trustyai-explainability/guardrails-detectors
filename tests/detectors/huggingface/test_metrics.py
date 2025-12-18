import os
import time
from unittest.mock import Mock

import pytest
import torch
from starlette.testclient import TestClient
from prometheus_client import REGISTRY

# DO NOT IMPORT THIS VALUE, if we import common.app before the test fixtures we can break prometheus multiprocessing
METRIC_PREFIX = "trustyai_guardrails"

def send_request(client: TestClient, detect: bool, slow: bool = False):
    payload = {
            "contents": ["this message is too long and should induce a detection from the model" if detect else "fine"],
            "detector_params": {"regex": []}
    }
    if slow:
        payload["contents"][0] = " ".join(payload["contents"]*1000)

    expected_status_code = 200
    response = client.post("/api/v1/text/contents", json=payload)
    if response.status_code != expected_status_code:
        print(response.text)
    assert response.status_code == expected_status_code


def get_metric_dict(client: TestClient):
    # In test mode with TestClient, we're running in a single process,
    # so multiprocess mode doesn't work. Use the default REGISTRY directly.
    from prometheus_client import generate_latest, REGISTRY
    metrics = generate_latest(REGISTRY).decode().split("\n")
    metric_dict = {}

    for m in metrics:
        if "trustyai" in m and "{" in m:
            key, value = m.split(" ")
            metric_dict[key] = float(value)

    return metric_dict

@pytest.fixture(scope="session")
def client(prometheus_multiproc_dir):
    # Clear any existing metrics from the REGISTRY before importing the app
    # This is needed because even in multiprocess mode, metrics are registered to REGISTRY
    collectors_to_unregister = [
        c for c in list(REGISTRY._collector_to_names.keys())
        if hasattr(c, '_name') and 'trustyai_guardrails' in c._name
    ]
    for collector in collectors_to_unregister:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    os.environ["MODEL_DIR"] = os.path.join(parent_dir, "dummy_models", "bert/BertForSequenceClassification")

    from detectors.huggingface.app import app
    from detectors.huggingface.detector import Detector
    detector = Detector()

    # patch the model to allow for control over detections - long messages will flag
    def detection_fn(*args, **kwargs):
        output = Mock()
        if kwargs["input_ids"].shape[-1] > 10:
            output.logits = torch.tensor([[0.0, 1.0]])
        else:
            output.logits = torch.tensor([[1.0, 0.0]])

        if kwargs["input_ids"].shape[-1] > 100:
            time.sleep(.25)
        return output

    class ModelMock:
        def __init__(self):
            self.config = Mock()
            self.config.id2label = detector.model.config.id2label
            self.config.problem_type = detector.model.config.problem_type
        def __call__(self, *args, **kwargs):
            return detection_fn(*args, **kwargs)

    detector.model = ModelMock()
    app.set_detector(detector, detector.registry_name)
    detector.set_instruments(app.state.instruments)
    return TestClient(app)

class TestMetrics:



    def test_prometheus(self, client: TestClient):
        for i in range(20):
            send_request(client=client, detect=i%3==0)

        expected_results = {
            f'{METRIC_PREFIX}_detections_total{{detector_kind="sequence_classifier",detector_name="BertForSequenceClassification"}}': 7.0,
            f'{METRIC_PREFIX}_errors_total{{detector_kind="sequence_classifier",detector_name="BertForSequenceClassification"}}': 0.0,
            f'{METRIC_PREFIX}_requests_total{{detector_kind="sequence_classifier",detector_name="BertForSequenceClassification"}}': 20.0,
        }

        metric_dict = get_metric_dict(client)

        for expected_key, expected_val in expected_results.items():
            assert expected_key in metric_dict, f"expected key {expected_key} not found in metric dict"
            assert metric_dict[expected_key] == expected_val,  f"metric {expected_key} value={metric_dict[expected_key]} did not match expected value {expected_val}"


    def test_runtime_metrics(self, client: TestClient):
        # 8 calls of this function should induce ~ 2 seconds of latency
        for _ in range(8):
            send_request(client=client, detect=False, slow=True)
        metric_dict = get_metric_dict(client)

        func_runtime = metric_dict[f'{METRIC_PREFIX}_runtime_total{{detector_kind="sequence_classifier",detector_name="BertForSequenceClassification"}}']
        assert func_runtime > 1.8
        assert func_runtime < 2.2