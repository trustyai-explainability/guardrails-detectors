# third-party imports
import os
import pytest
import torch
from unittest.mock import Mock, patch

# relative imports
from guardrails_detectors.huggingface.detector import Detector, ContentAnalysisResponse
from guardrails_detectors.huggingface.scheme import ContentAnalysisHttpRequest


@pytest.fixture
def setup_environment():
    """
    Setup the required environment variable for the model directory.
    """
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    os.environ["MODEL_DIR"] = os.path.join(parent_dir, "dummy_models")


class TestDetectorRun:
    @pytest.fixture(autouse=True)
    def setup(self, setup_environment):
        pass

    @pytest.fixture
    def detector_sequence(self):
        model_dir = os.path.join(
            os.environ["MODEL_DIR"], "bert/BertForSequenceClassification"
        )
        os.environ["MODEL_DIR"] = model_dir
        return Detector()

    @pytest.fixture
    def detector_causal_lm(self):
        with patch.dict("os.environ", {"MODEL_DIR": "/dummy/path"}):
            detector = Detector.__new__(Detector)

            # Mock process_causal_lm method
            detector.process_causal_lm = Mock(
                return_value=[
                    ContentAnalysisResponse(
                        start=0,
                        end=12,
                        detection="causal_lm",
                        detection_type="causal_lm",
                        score=0.8,
                        sequence_classification="harm",
                        sequence_probability=0.8,
                        token_classifications=None,
                        token_probabilities=None,
                        text="Test content",
                        evidences=[],
                    )
                ]
            )

            detector.model_name = "causal_lm"
            detector.is_causal_lm = True
            detector.is_sequence_classifier = False
            detector.risk_names = ["harm", "bias"]

            return detector

    def test_run_sequence_classifier_single_short_input(self, detector_sequence):
        request = ContentAnalysisHttpRequest(contents=["Test content"])
        results = detector_sequence.run(request)

        assert len(results) == 1
        assert isinstance(results[0][0], ContentAnalysisResponse)
        assert results[0][0].detection_type == "sequence_classification"

    def test_run_sequence_classifier_single_long_input(self, detector_sequence):
        request = ContentAnalysisHttpRequest(
            contents=[
                "This is a long content. " * 1_000,
            ]
        )
        results = detector_sequence.run(request)

        assert len(results) == 1
        assert isinstance(results[0][0], ContentAnalysisResponse)
        assert results[0][0].detection_type == "sequence_classification"

    def test_run_sequence_classifier_empty_input(self, detector_sequence):
        request = ContentAnalysisHttpRequest(contents=[""])
        results = detector_sequence.run(request)

        assert len(results) == 1
        assert isinstance(results[0][0], ContentAnalysisResponse)
        assert results[0][0].detection_type == "sequence_classification"

    def test_run_sequence_classifier_multiple_contents(self, detector_sequence):
        request = ContentAnalysisHttpRequest(contents=["Content 1", "Content 2"])
        results = detector_sequence.run(request)

        assert len(results) == 2
        for content_analysis in results:
            assert len(content_analysis) == 1
            assert isinstance(content_analysis[0], ContentAnalysisResponse)
            assert content_analysis[0].detection_type == "sequence_classification"

    def test_run_unsupported_model(self):
        detector = Detector.__new__(Detector)
        detector.is_causal_lm = False
        detector.is_sequence_classifier = False

        request = ContentAnalysisHttpRequest(contents=["Test content"])
        with pytest.raises(ValueError, match="Unsupported model type for analysis"):
            detector.run(request)

    def test_run_causal_lm_single_short_input(self, detector_causal_lm):
        request = ContentAnalysisHttpRequest(contents=["Test content"])
        results = detector_causal_lm.run(request)

        assert len(results) == 1
        assert isinstance(results[0][0], ContentAnalysisResponse)
        assert results[0][0].detection_type == "causal_lm"

    def test_run_causal_lm_single_long_input(self, detector_causal_lm):
        request = ContentAnalysisHttpRequest(
            contents=[
                "This is a long content. " * 1_000,
            ]
        )
        results = detector_causal_lm.run(request)

        assert len(results) == 1
        assert isinstance(results[0][0], ContentAnalysisResponse)
        assert results[0][0].detection_type == "causal_lm"

    def test_run_causal_lm_empty_input(self, detector_causal_lm):
        request = ContentAnalysisHttpRequest(contents=[""])
        results = detector_causal_lm.run(request)

        assert len(results) == 1
        assert isinstance(results[0][0], ContentAnalysisResponse)
        assert results[0][0].detection_type == "causal_lm"

    def tes_run_causal_lm_multiple_contents(self, detector_causal_lm):
        request = ContentAnalysisHttpRequest(contents=["Content 1", "Content 2"])
        results = detector_causal_lm.run(request)

        assert len(results) == 2
        for content_analysis in results:
            assert len(content_analysis) == 1
            assert isinstance(content_analysis[0], ContentAnalysisResponse)
            assert content_analysis[0].detection_type == "causal_lm"
