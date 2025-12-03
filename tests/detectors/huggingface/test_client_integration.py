"""
Integration tests for HF detector FastAPI lifespan context
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Set up paths and MODEL_DIR before app import
_current_dir = os.path.dirname(__file__)
_tests_dir = os.path.dirname(os.path.dirname(_current_dir))
_project_root = os.path.dirname(_tests_dir)
_detectors_path = os.path.join(_project_root, "detectors")
_huggingface_path = os.path.join(_detectors_path, "huggingface")

for path in [_huggingface_path, _detectors_path, _project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ["MODEL_DIR"] = os.path.join(
    _tests_dir, "dummy_models", "bert/BertForSequenceClassification"
)

from app import app  # noqa: E402


class TestLifespanIntegration:
    """Test FastAPI lifespan context for HF detector."""

    @pytest.fixture
    def client(self):
        """Create test client with lifespan-initialized detector."""
        with TestClient(app) as test_client:
            yield test_client

    def test_lifespan_loads_detector(self, client):
        """Verify lifespan initializes detector on startup."""
        detectors = app.get_all_detectors()
        assert len(detectors) > 0
        detector = list(detectors.values())[0]
        assert detector.model is not None
        assert detector.tokenizer is not None

    def test_lifespan_handles_requests(self, client):
        """Verify requests work through lifespan-initialized app."""
        response = client.post(
            "/api/v1/text/contents",
            json={"contents": ["Test message"], "detector_params": {}},
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_multiple_requests(self, client):
        """Verify detector handles multiple requests without state leakage."""
        for i in range(10):
            response = client.post(
                "/api/v1/text/contents",
                json={"contents": [f"Request {i}"], "detector_params": {}},
            )
            assert response.status_code == 200
            assert isinstance(response.json(), list)

    def test_lifespan_cleanup(self, client):
        """Verify lifespan cleanup runs on shutdown."""
        assert len(app.get_all_detectors()) > 0

        client.__exit__(None, None, None)

        assert len(app.get_all_detectors()) == 0
