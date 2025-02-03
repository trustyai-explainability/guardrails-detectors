# third-party imports
import numpy as np
import pytest
import torch
from unittest.mock import Mock, patch
from transformers import PreTrainedTokenizer

# relative imports
from detectors.huggingface.detector import Detector, ContentAnalysisResponse


class TestDetectorParseOutput:
    @pytest.fixture
    def detector(self):
        detector = Detector.__new__(Detector)
        detector.tokenizer = Mock(spec=PreTrainedTokenizer)
        detector.get_probabilities = Mock()
        return detector

    @pytest.fixture
    def mock_output(self):
        output = Mock()
        output.sequences = torch.tensor([[1, 2, 3, 4]])
        output.scores = [torch.randn(1, 5)]
        return output

    @pytest.fixture
    def default_params(self):
        return {
            "input_len": 2,
            "nlogprobs": 5,
            "safe_token": "Safe",
            "unsafe_token": "Unsafe",
        }

    def test_parse_output_safe_classification(
        self, detector, mock_output, default_params
    ):
        """Test safe token classification with probabilities"""
        detector.tokenizer.decode.return_value = "safe"
        detector.get_probabilities.return_value = torch.tensor([0.7, 0.3])

        label, prob = detector.parse_output(output=mock_output, **default_params)

        assert label == "Safe"
        assert isinstance(prob, float)
        np.testing.assert_almost_equal(prob, 0.3, decimal=5)

    def test_parse_output_unsafe_classification(
        self, detector, mock_output, default_params
    ):
        """Test unsafe token classification with probabilities"""
        detector.tokenizer.decode.return_value = "unsafe"
        detector.get_probabilities.return_value = torch.tensor([0.3, 0.7])

        label, prob = detector.parse_output(output=mock_output, **default_params)

        assert label == "Unsafe"
        assert isinstance(prob, float)
        np.testing.assert_almost_equal(prob, 0.7, decimal=5)

    def test_parse_output_failed_classification(
        self, detector, mock_output, default_params
    ):
        """Test when decoded token doesn't match safe/unsafe"""
        detector.tokenizer.decode.return_value = "invalid"
        detector.get_probabilities.return_value = torch.tensor([0.5, 0.5])

        label, prob = detector.parse_output(output=mock_output, **default_params)

        assert label == "failed"
        assert prob == 0.5

    def test_parse_output_empty_sequence(self, detector, default_params):
        """Test with empty sequence"""
        mock_output = Mock()
        mock_output.sequences = torch.tensor([[]])
        detector.tokenizer.decode.return_value = ""

        label, prob = detector.parse_output(
            output=mock_output,
            input_len=0,
            nlogprobs=0,
            safe_token="Safe",
            unsafe_token="Unsafe",
        )

        assert label == "failed"
        assert prob is None
