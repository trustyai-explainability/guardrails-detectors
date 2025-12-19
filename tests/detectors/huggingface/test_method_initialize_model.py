# third-party imports
import os
import pytest

# local imports
from detectors.common.scheme import ContentAnalysisResponse
from detectors.huggingface.detector import Detector


@pytest.fixture
def setup_environment():
    """
    Setup the required environment variable for the model directory.
    """
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    os.environ["MODEL_DIR"] = os.path.join(parent_dir, "dummy_models")


# tests to check the model initialization
class TestDetector:
    @pytest.fixture(autouse=True)
    def setup(self, setup_environment):
        pass

    def test_initialisation_without_model_dir(self):
        os.environ.pop("MODEL_DIR", None)
        with pytest.raises(
            ValueError, match="MODEL_DIR environment variable is not set"
        ):
            Detector()

    def test_initialisation_with_unusual_task(self):
        model_dir = os.path.join(os.environ["MODEL_DIR"], "bert/BertModel")
        os.environ["MODEL_DIR"] = model_dir
        with pytest.raises(ValueError, match="Unsupported model architecture."):
            Detector()

    def test_initialisation_with_token_classifier(self):
        model_dir = os.path.join(
            os.environ["MODEL_DIR"], "bert/BertForTokenClassification"
        )
        os.environ["MODEL_DIR"] = model_dir
        with pytest.raises(
            ValueError, match="Token classification models are not supported."
        ):
            Detector()

    def test_initialization_with_sequence_classifier(self):
        model_dir = os.path.join(
            os.environ["MODEL_DIR"], "bert/BertForSequenceClassification"
        )
        os.environ["MODEL_DIR"] = model_dir
        detector = Detector()
        assert detector.model_name == "sequence_classifier"
        assert detector.is_sequence_classifier is True
        assert detector.is_causal_lm is False

    def test_intialisation_with_casual_lm_but_no_granite(self):
        model_dir = os.path.join(os.environ["MODEL_DIR"], "gpt2/GPT2Model")
        os.environ["MODEL_DIR"] = model_dir
        with pytest.raises(ValueError, match="Unsupported model architecture."):
            Detector()
