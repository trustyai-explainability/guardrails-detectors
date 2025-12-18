# third-party imports
import os
import pytest
import torch
from unittest.mock import Mock, patch

# relative imports
from detectors.huggingface.detector import Detector, ContentAnalysisResponse


class MockGraniteOutput:
    def __init__(self):
        self.sequences = torch.tensor([[1, 2, 3, 4]])
        self.scores = [torch.randn(1, 5)]


@pytest.fixture
def setup_environment():
    """Setup the required environment variable for the model directory."""
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    os.environ["MODEL_DIR"] = os.path.join(parent_dir, "dummy_models")


class TestDetector:
    @pytest.fixture(autouse=True)
    def setup(self, setup_environment):
        pass

    @pytest.fixture
    def detector_instance(self):
        with patch.dict("os.environ", {"MODEL_DIR": "/dummy/path"}):
            detector = Detector.__new__(Detector)

            detector.tokenizer = Mock()
            detector.tokenizer.apply_chat_template = Mock(
                return_value=torch.tensor([[1, 2, 3]])
            )
            detector.tokenizer.decode = Mock(return_value="Yes")

            detector.model = Mock()
            detector.model.device = torch.device("cpu")
            detector.model.generate = Mock(return_value=MockGraniteOutput())

            detector.model_name = "causal_lm"
            detector.is_causal_lm = True
            detector.cuda_device = None
            detector.risk_names = ["harm", "bias"]

            return detector

    def validate_results(self, results, input_text, detector):
        """Helper method to validate the classification results"""
        assert len(results) == len(detector.risk_names)

        for result in results:
            expected_fields = [
                "start",
                "end",
                "detection",
                "detection_type",
                "score",
                "text",
                "evidences",
                "metadata",
            ]

            for field in expected_fields:
                assert hasattr(
                    result, field
                ), f"Missing '{field}' in ContentAnalysisResponse"

            assert isinstance(result, ContentAnalysisResponse)
            assert isinstance(result.start, int)
            assert isinstance(result.end, int)
            assert isinstance(result.detection, str)
            assert isinstance(result.detection_type, str)
            assert isinstance(result.score, float)
            assert isinstance(result.text, str)
            assert isinstance(result.evidences, list)

            assert 0 <= result.start <= len(input_text)
            assert 0 <= result.end <= len(input_text)
            assert 0.0 <= result.score <= 1.0

    def test_process_causal_lm_single_short_input(self, detector_instance):
        text = "This is a test."
        results = detector_instance.process_causal_lm(text)
        self.validate_results(results, text, detector_instance)

    def test_process_causal_lm_single_long_input(self, detector_instance):
        text = "This is a test." * 1_000
        results = detector_instance.process_causal_lm(text)
        self.validate_results(results, text, detector_instance)

    def test_process_causal_lm_single_empty_input(self, detector_instance):
        text = ""
        results = detector_instance.process_causal_lm(text)
        self.validate_results(results, text, detector_instance)
