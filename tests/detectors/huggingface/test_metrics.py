import random
import os
import time
from collections import namedtuple
from unittest import mock
from unittest.mock import Mock, MagicMock

import pytest
import torch
from starlette.testclient import TestClient

from detectors.huggingface.detector import Detector
from detectors.huggingface.app import app


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
    metrics = client.get("/metrics")
    metrics = metrics.content.decode().split("\n")
    metric_dict = {}

    for m in metrics:
        if "trustyai" in m and "{" in m:
            key, value = m.split(" ")
            metric_dict[key] = float(value)

    return metric_dict

class TestMetrics:
    @pytest.fixture
    def client(self):
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        os.environ["MODEL_DIR"] = os.path.join(parent_dir, "dummy_models", "bert/BertForSequenceClassification")

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
        detector.add_instruments(app.state.instruments)
        return TestClient(app)



    def test_prometheus(self, client: TestClient):
        for i in range(20):
            send_request(client=client, detect=i%3==0)

        expected_results = {
            'trustyai_guardrails_detections_total{detector_kind="sequence_classifier",detector_name="BertForSequenceClassification"}': 7.0,
            'trustyai_guardrails_errors_total{detector_kind="sequence_classifier",detector_name="BertForSequenceClassification"}': 0.0,
            'trustyai_guardrails_requests_total{detector_kind="sequence_classifier",detector_name="BertForSequenceClassification"}': 20.0,
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

        func_runtime = metric_dict['trustyai_guardrails_runtime_total{detector_kind="sequence_classifier",detector_name="BertForSequenceClassification"}']
        assert func_runtime > 1.8
        assert func_runtime < 2.2