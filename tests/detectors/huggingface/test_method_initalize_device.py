# third-party imports
import os
import pytest
import torch
from unittest.mock import patch, MagicMock

# relative imports
from detectors.huggingface.detector import Detector


# tests to check the device initialization
class TestDetectorDevice:
    @pytest.fixture
    def detector(self):
        detector = Detector.__new__(Detector)
        detector.model = MagicMock()
        return detector

    @pytest.mark.parametrize("cuda_available", [False])
    @patch("detectors.huggingface.detector.logger")
    def test_initialize_device_cpu_only(self, mock_logger, detector, cuda_available):
        """CUDA is not available"""
        with patch("torch.cuda.is_available", return_value=cuda_available):
            detector.initialize_device()

            assert detector.cuda_device == torch.device("cpu")
            mock_logger.info.assert_called_once_with(
                "CUDA is not available. Using CPU."
            )

    @pytest.mark.parametrize("cuda_available", [True])
    @patch("detectors.huggingface.detector.logger")
    def test_initialize_device_cuda_available(
        self, mock_logger, detector, cuda_available
    ):
        """CUDA is available"""
        with patch("torch.cuda.is_available", return_value=cuda_available):
            detector.initialize_device()

            assert detector.cuda_device == torch.device("cuda")
            assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:512"
            mock_logger.info.assert_called_once_with(
                f"CUDA device initialized: {torch.device('cuda')}"
            )
