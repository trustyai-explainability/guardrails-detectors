# third-party imports
import pytest
import torch
from unittest.mock import Mock
from transformers import PreTrainedTokenizer

# relative imports
from guardrails_detectors.huggingface.detector import Detector


class TestGetProbabilities:
    @pytest.fixture
    def detector(self):
        detector = Detector.__new__(Detector)
        detector.tokenizer = Mock(spec=PreTrainedTokenizer)
        return detector

    def test_normal_case(self, detector):
        # Setup
        logprobs = [
            Mock(values=torch.tensor([[0.0, -1.0]]), indices=torch.tensor([[1, 2]]))
        ]
        detector.tokenizer.convert_ids_to_tokens.side_effect = lambda x: (
            "safe" if x == 1 else "unsafe"
        )
        result = detector.get_probabilities(logprobs, "safe", "unsafe")
        assert isinstance(result, torch.Tensor)
        assert len(result) == 2
        assert torch.allclose(result.sum(), torch.tensor(1.0))
        assert result[0] > result[1]  # Safe token has higher probability

    def test_empty_logprobs(self, detector):
        result = detector.get_probabilities([], "safe", "unsafe")
        assert torch.allclose(result, torch.tensor([0.5, 0.5]))

    def test_very_small_probabilities(self, detector):
        logprobs = [
            Mock(values=torch.tensor([[-50.0, -50.0]]), indices=torch.tensor([[1, 2]]))
        ]
        detector.tokenizer.convert_ids_to_tokens.side_effect = lambda x: (
            "safe" if x == 1 else "unsafe"
        )
        result = detector.get_probabilities(logprobs, "safe", "unsafe")
        assert torch.allclose(result.sum(), torch.tensor(1.0))
        assert torch.allclose(result[0], result[1])  # Should be equal probabilities

    def test_case_sensitivity(self, detector):
        logprobs = [Mock(values=torch.tensor([[0.0]]), indices=torch.tensor([[1]]))]
        detector.tokenizer.convert_ids_to_tokens.return_value = "SAFE"
        result = detector.get_probabilities(logprobs, "safe", "unsafe")
        assert result[0] > result[1]

    def test_invalid_tokens(self, detector):
        logprobs = [Mock(values=torch.tensor([[0.0]]), indices=torch.tensor([[1]]))]
        detector.tokenizer.convert_ids_to_tokens.return_value = "invalid"
        result = detector.get_probabilities(logprobs, "safe", "unsafe")
        assert torch.allclose(result, torch.tensor([0.5, 0.5]))
