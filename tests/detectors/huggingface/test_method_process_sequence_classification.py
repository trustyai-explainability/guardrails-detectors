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

    def test_default_max_length(self, detector_instance):
        os.environ.pop("MAX_LENGTH", None)
        assert detector_instance.default_max_length == 512

    def test_env_max_length_override(self, setup_environment):
        model_dir = os.path.join(
            os.environ["MODEL_DIR"], "bert/BertForSequenceClassification"
        )
        os.environ["MODEL_DIR"] = model_dir
        os.environ["MAX_LENGTH"] = "256"
        try:
            detector = Detector()
            assert detector.default_max_length == 256
        finally:
            os.environ.pop("MAX_LENGTH", None)

    def test_invalid_env_max_length_falls_back_to_default(self, setup_environment):
        model_dir = os.path.join(
            os.environ["MODEL_DIR"], "bert/BertForSequenceClassification"
        )
        os.environ["MODEL_DIR"] = model_dir
        os.environ["MAX_LENGTH"] = "not_a_number"
        try:
            detector = Detector()
            assert detector.default_max_length == 512
        finally:
            os.environ.pop("MAX_LENGTH", None)

    def test_env_max_length_clamped_to_model_max(self, setup_environment):
        """MAX_LENGTH exceeding model capacity should be clamped in constructor."""
        model_dir = os.path.join(
            os.environ["MODEL_DIR"], "bert/BertForSequenceClassification"
        )
        os.environ["MODEL_DIR"] = model_dir
        os.environ["MAX_LENGTH"] = "999999"
        try:
            detector = Detector()
            assert detector.default_max_length == detector.tokenizer.model_max_length
        finally:
            os.environ.pop("MAX_LENGTH", None)

    def test_request_max_length_override(self, detector_instance):
        text = "This is a test."
        results = detector_instance.process_sequence_classification(
            text, detector_params={"max_length": 64}
        )
        self.validate_results(results, text)

    def test_max_length_clamped_to_model_max(self, detector_instance):
        """Requesting max_length beyond model capacity should clamp to model max."""
        model_max = detector_instance.tokenizer.model_max_length
        params = detector_instance._resolve_params({"max_length": model_max + 1000})
        assert params.max_length == model_max

    def test_invalid_request_max_length_uses_default(self, detector_instance):
        text = "This is a test."
        for invalid in ["abc", -1, 0, 3.5, True, False, [], {}]:
            results = detector_instance.process_sequence_classification(
                text, detector_params={"max_length": invalid}
            )
            self.validate_results(results, text)

    def test_invalid_threshold_in_detector_params(self, detector_instance):
        """Non-numeric threshold should fall back to default without error."""
        text = "This is a test."
        for invalid in ["high", True, [], {}]:
            results = detector_instance.process_sequence_classification(
                text, detector_params={"threshold": invalid}
            )
            self.validate_results(results, text)

    def test_invalid_label_thresholds_in_detector_params(self, detector_instance):
        """Malformed label_thresholds should be ignored gracefully."""
        text = "This is a test."
        results = detector_instance.process_sequence_classification(
            text, detector_params={"label_thresholds": "not a dict"}
        )
        self.validate_results(results, text)

    def test_threshold_affects_detection_count(self, detector_instance):
        """Valid thresholds should change which detections are returned."""
        text = "This is a test."
        results_all = detector_instance.process_sequence_classification(
            text, threshold=0.0
        )
        results_none = detector_instance.process_sequence_classification(
            text, threshold=1.0
        )
        assert len(results_none) <= len(results_all)

    def test_safe_labels_bare_string_normalised(self, detector_instance):
        """A bare string should be treated as a single-element list."""
        params = detector_instance._resolve_params({"safe_labels": "LABEL_0"})
        assert "LABEL_0" in params.safe_labels
