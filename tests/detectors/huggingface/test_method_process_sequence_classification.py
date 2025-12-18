# third-party imports
import os
import pytest

# relative imports
from detectors.huggingface.detector import Detector, ContentAnalysisResponse


@pytest.fixture
def setup_environment():
    """
    Setup the required environment variable for the model directory.
    """
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    os.environ["MODEL_DIR"] = os.path.join(parent_dir, "dummy_models")


# tests to check the detector output
class TestDetector:
    @pytest.fixture(autouse=True)
    def setup(self, setup_environment):
        pass

    @pytest.fixture
    def detector_instance(self):
        model_dir = os.path.join(
            os.environ["MODEL_DIR"], "bert/BertForSequenceClassification"
        )
        os.environ["MODEL_DIR"] = model_dir
        return Detector()

    def validate_results(self, results, input_text):
        """Helper method to validate the classification results"""
        assert len(results) == 1
        result = results[0]
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
        assert isinstance(result.start, int), "start should be integer"
        assert isinstance(result.end, int), "end should be integer"
        assert isinstance(result.detection, str), "detection should be string"
        assert isinstance(result.detection_type, str), "detection_type should be string"
        assert isinstance(result.score, float), "score should be float"
        assert isinstance(result.text, str), "text should be string"
        assert isinstance(result.evidences, list), "evidences should be list"

        assert (
            0 <= result.start <= len(input_text)
        ), "start should be within text bounds"
        assert 0 <= result.end <= len(input_text), "end should be within text bounds"
        assert 0.0 <= result.score <= 1.0, "score should be between 0 and 1"

        return result

    def test_process_sequence_classification_single_short_input(
        self, detector_instance
    ):
        text = "This is a test."
        results = detector_instance.process_sequence_classification(text)
        self.validate_results(results, text)

    def test_process_sequence_classification_single_long_input(self, detector_instance):
        text = "This is a test." * 1_000
        results = detector_instance.process_sequence_classification(text)
        self.validate_results(results, text)

    def test_process_sequence_classification_single_empty_input(
        self, detector_instance
    ):
        text = ""
        results = detector_instance.process_sequence_classification(text)
        self.validate_results(results, text)
