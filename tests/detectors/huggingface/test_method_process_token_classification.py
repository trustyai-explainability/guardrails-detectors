# third-party imports
import json
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


class TestTokenClassification:
    @pytest.fixture(autouse=True)
    def setup(self, setup_environment):
        pass

    @pytest.fixture
    def detector_instance(self):
        model_dir = os.path.join(
            os.environ["MODEL_DIR"], "bert/BertForTokenClassification"
        )
        os.environ["MODEL_DIR"] = model_dir
        # Clear SAFE_LABELS so the token classifier default ["O"] is used
        os.environ.pop("SAFE_LABELS", None)
        return Detector()

    def test_model_type_detected(self, detector_instance):
        assert detector_instance.is_token_classifier is True
        assert detector_instance.is_sequence_classifier is False
        assert detector_instance.is_causal_lm is False
        assert detector_instance.model_name == "token_classifier"

    def test_default_safe_labels(self, detector_instance):
        assert detector_instance.safe_labels == ["O"]

    def test_env_safe_labels_override(self, setup_environment):
        model_dir = os.path.join(
            os.environ["MODEL_DIR"], "bert/BertForTokenClassification"
        )
        os.environ["MODEL_DIR"] = model_dir
        os.environ["SAFE_LABELS"] = '["LABEL_0", "LABEL_1"]'
        try:
            detector = Detector()
            assert detector.safe_labels == ["LABEL_0", "LABEL_1"]
        finally:
            os.environ.pop("SAFE_LABELS", None)

    def test_malformed_safe_labels_falls_back_to_O(self, setup_environment):
        """Malformed SAFE_LABELS should fall back to ["O"] for token classifiers, not [0]."""
        model_dir = os.path.join(
            os.environ["MODEL_DIR"], "bert/BertForTokenClassification"
        )
        os.environ["MODEL_DIR"] = model_dir
        os.environ["SAFE_LABELS"] = "not valid json"
        try:
            detector = Detector()
            assert detector.safe_labels == ["O"]
        finally:
            os.environ.pop("SAFE_LABELS", None)

    def test_integer_safe_labels_filter_by_index(self, setup_environment):
        """Integer safe labels from env should match label indices."""
        model_dir = os.path.join(
            os.environ["MODEL_DIR"], "bert/BertForTokenClassification"
        )
        os.environ["MODEL_DIR"] = model_dir
        os.environ.pop("SAFE_LABELS", None)
        # First create detector to discover actual label count
        detector = Detector()
        all_indices = list(detector.model.config.id2label.keys())
        os.environ["SAFE_LABELS"] = json.dumps(all_indices)
        try:
            detector = Detector()
            results = detector.process_token_classification(
                "This is a test.", threshold=0.0
            )
            assert results == []
        finally:
            os.environ.pop("SAFE_LABELS", None)

    def test_env_and_request_safe_labels_union(self, setup_environment):
        """Both env and request safe_labels should be merged via union."""
        model_dir = os.path.join(
            os.environ["MODEL_DIR"], "bert/BertForTokenClassification"
        )
        os.environ["MODEL_DIR"] = model_dir
        os.environ["SAFE_LABELS"] = '["O"]'
        try:
            detector = Detector()
            text = "This is a test."
            all_labels = list(detector.model.config.id2label.values())
            # Request safe_labels covers all labels except "O" (already in env)
            request_safe = [l for l in all_labels if l != "O"]
            results = detector.process_token_classification(
                text, detector_params={"safe_labels": request_safe}, threshold=0.0
            )
            # Union of env + request covers all labels, so nothing should be detected
            assert results == []
        finally:
            os.environ.pop("SAFE_LABELS", None)

    def test_process_single_short_input(self, detector_instance):
        text = "This is a test."
        results = detector_instance.process_token_classification(text)
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, ContentAnalysisResponse)
            assert 0 <= result.start <= len(text)
            assert 0 <= result.end <= len(text)
            assert result.start < result.end
            assert 0.0 <= result.score <= 1.0
            # text field should be the detected span, not the whole input
            assert result.text == text[result.start:result.end]
            assert isinstance(result.detection_type, str)

    def test_process_single_long_input(self, detector_instance):
        text = "This is a test. " * 1_000
        results = detector_instance.process_token_classification(text)
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, ContentAnalysisResponse)
            assert result.start < result.end

    def test_process_empty_input(self, detector_instance):
        text = ""
        results = detector_instance.process_token_classification(text)
        assert isinstance(results, list)

    def test_all_labels_safe_by_name_returns_empty(self, detector_instance):
        text = "This is a test."
        # Mark all possible labels as safe by name
        all_labels = list(detector_instance.model.config.id2label.values())
        results = detector_instance.process_token_classification(
            text, detector_params={"safe_labels": all_labels}
        )
        assert results == []

    def test_all_labels_safe_by_index_returns_empty(self, detector_instance):
        text = "This is a test."
        # Mark all possible labels as safe by index
        all_indices = list(detector_instance.model.config.id2label.keys())
        results = detector_instance.process_token_classification(
            text, detector_params={"safe_labels": all_indices}
        )
        assert results == []

    def test_threshold_filtering(self, detector_instance):
        text = "This is a test."
        # With threshold=1.0, no token should have probability == 1.0
        results_high = detector_instance.process_token_classification(
            text, threshold=1.0
        )
        # With threshold=0.0, all non-safe tokens should be detected
        results_low = detector_instance.process_token_classification(
            text, threshold=0.0
        )
        assert len(results_high) <= len(results_low)

    def test_no_overlapping_spans(self, detector_instance):
        """At low thresholds, each token should appear in at most one span."""
        text = "John Smith lives in New York."
        results = detector_instance.process_token_classification(
            text, threshold=0.0
        )
        # Check that no character position is covered by more than one span
        for i, a in enumerate(results):
            for b in results[i + 1:]:
                assert a.end <= b.start or b.end <= a.start, (
                    f"Overlapping spans: [{a.start}:{a.end}] and [{b.start}:{b.end}]"
                )

    def test_run_routes_to_token_classification(self, detector_instance):
        from detectors.common.scheme import ContentAnalysisHttpRequest
        request = ContentAnalysisHttpRequest(
            contents=["Test content"], detector_params=None
        )
        results = detector_instance.run(request)
        assert len(results) == 1
        assert isinstance(results[0], list)

