import pytest
import os
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Tuple

# Import the detector components
from detectors.llm_judge.detector import LLMJudgeDetector
from detectors.llm_judge.scheme import (
    ContentAnalysisHttpRequest,
    ContentAnalysisResponse,
    GenerationAnalysisHttpRequest,
    GenerationAnalysisResponse,
)

# Import vLLM Judge components for mocking
from vllm_judge import EvaluationResult
from vllm_judge.exceptions import MetricNotFoundError


class TestLLMJudgeDetectorContentAnalysis:
    """Test suite for LLMJudgeDetector content analysis."""
    
    @pytest.fixture
    def mock_judge_result(self) -> EvaluationResult:
        """Mock EvaluationResult for testing."""
        return EvaluationResult(
            decision="SAFE",
            reasoning="This content appears to be safe with no concerning elements.",
            score=0.9,
            metadata={"model": "test-model"}
        )
    
    @pytest.fixture
    def detector_with_mock_judge(self, mock_judge_result) -> Tuple[LLMJudgeDetector, AsyncMock]:
        """Create detector with mocked Judge."""
        with patch.dict(os.environ, {"VLLM_BASE_URL": "http://test:8000"}):
            with patch('vllm_judge.Judge.from_url') as mock_judge_class:
                # Create mock judge instance
                mock_judge_instance = AsyncMock()
                mock_judge_instance.evaluate = AsyncMock(return_value=mock_judge_result)
                mock_judge_instance.config.model = "test-model"
                mock_judge_instance.config.base_url = "http://test:8000"
                mock_judge_instance.close = AsyncMock()
                
                mock_judge_class.return_value = mock_judge_instance
                
                detector = LLMJudgeDetector()
                return detector, mock_judge_instance

    def test_detector_initialization_success(self) -> None:
        """Test successful detector initialization."""
        with patch.dict(os.environ, {"VLLM_BASE_URL": "http://test:8000"}):
            with patch('vllm_judge.Judge.from_url') as mock_judge:
                mock_instance = Mock()
                mock_instance.config.model = "test-model"
                mock_instance.config.base_url = "http://test:8000"
                mock_judge.return_value = mock_instance
                
                detector = LLMJudgeDetector()
                
                assert detector.vllm_base_url == "http://test:8000"
                assert detector.judge is not None
                mock_judge.assert_called_once_with(base_url="http://test:8000")

    def test_detector_initialization_missing_url(self) -> None:
        """Test detector initialization fails without VLLM_BASE_URL."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="VLLM_BASE_URL environment variable is required"):
                LLMJudgeDetector()
    
    def test_detector_initialization_unreachable_url(self) -> None:
        """Test detector initialization fails with unreachable URL."""
        with patch.dict(os.environ, {"VLLM_BASE_URL": "http://unreachable:8000"}):
            with pytest.raises(Exception, match="Failed to detect model"):
                LLMJudgeDetector()

    def test_close_detector(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test closing the detector properly closes the judge."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        asyncio.run(detector.close())
        
        mock_judge.close.assert_called_once()

    def test_evaluate_single_content_basic_metric(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test basic evaluation with just a metric."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        params = {"metric": "safety"}
        result = asyncio.run(detector.evaluate_single_content("Test content", params))
        
        # Verify judge.evaluate was called correctly
        mock_judge.evaluate.assert_called_once_with(
            content="Test content",
            metric="safety"
        )
        
        # Verify response format
        assert isinstance(result, ContentAnalysisResponse)
        assert result.detection == "SAFE"
        assert result.score == 0.9
        assert result.text == "Test content"
        assert result.detection_type == "llm_judge"
        assert "reasoning" in result.metadata

    def test_evaluate_single_content_full_parameters(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test evaluation with all vLLM Judge parameters."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        params = {
            "input": "What is AI?",
            "criteria": "accuracy and completeness", 
            "rubric": "Score based on factual accuracy",
            "scale": [1, 10],
            "examples": [{"input": "test", "output": "example"}],
            "system_prompt": "You are an expert evaluator",
            "context": "This is a technical discussion about {subject}",
            "template_vars": {"subject": "artificial intelligence"}
        }
        content = "AI (Artificial Intelligence) refers to the simulation of human intelligence in machines that are programmed to"
        " think, learn, and perform tasks typically requiring human intelligence."
        asyncio.run(detector.evaluate_single_content(content, params))
        
        # Verify all parameters were passed through
        expected_call = {
            "content": content,
            **params
        }
        mock_judge.evaluate.assert_called_once_with(**expected_call)

    def test_evaluate_single_content_criteria_without_metric(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test evaluation with criteria but no metric (should default scale)."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        params = {
            "criteria": "custom evaluation criteria",
            "rubric": "Custom rubric"
        }
        
        asyncio.run(detector.evaluate_single_content("Test content", params))
        
        # Should add default scale when criteria provided without metric
        expected_params = {
            "content": "Test content",
            "criteria": "custom evaluation criteria",
            "rubric": "Custom rubric",
            "scale": (0, 1)
        }
        mock_judge.evaluate.assert_called_once_with(**expected_params)

    def test_evaluate_single_content_no_params(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test evaluation with no parameters (should default to safety)."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        params = {}
        
        asyncio.run(detector.evaluate_single_content("Test content", params))
        
        # Should default to safety metric
        expected_params = {
            "content": "Test content",
            "metric": "safety"
        }
        mock_judge.evaluate.assert_called_once_with(**expected_params)

    def test_evaluate_single_content_invalid_metric(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test evaluation with invalid metric raises error."""
        detector: LLMJudgeDetector
        detector, _ = detector_with_mock_judge
        
        params = {"metric": "invalid_metric"}
        
        with pytest.raises(MetricNotFoundError, match="Metric 'invalid_metric' not found"):
            asyncio.run(detector.evaluate_single_content("Test content", params))

    def test_run_single_content(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test the run method with single content."""
        detector: LLMJudgeDetector
        detector, _ = detector_with_mock_judge
        
        request = ContentAnalysisHttpRequest(
            contents=["Test content"],
            detector_params={"metric": "safety"}
        )
        
        result = asyncio.run(detector.analyze_content(request))
        
        assert len(result) == 1
        assert len(result[0]) == 1
        assert isinstance(result[0][0], ContentAnalysisResponse)

        assert isinstance(result[0][0].text, str)
        assert result[0][0].text == "Test content"

        assert isinstance(result[0][0].score, float)
        assert result[0][0].score is not None

        assert isinstance(result[0][0].detection, str)
        assert result[0][0].detection is not None

        assert isinstance(result[0][0].detection_type, str)
        assert result[0][0].detection_type == "llm_judge"

        assert isinstance(result[0][0].metadata, dict)
        assert result[0][0].metadata is not None
        assert "reasoning" in result[0][0].metadata
        assert result[0][0].metadata["reasoning"] is not None

    def test_run_multiple_contents(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test the run method with multiple contents."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        request = ContentAnalysisHttpRequest(
            contents=["Content 1", "Content 2", "Content 3"],
            detector_params={"metric": "safety"}
        )
        
        result = asyncio.run(detector.analyze_content(request))
        
        assert len(result) == 3
        for i, analysis_list in enumerate(result):
            assert len(analysis_list) == 1
            assert analysis_list[0].text == f"Content {i+1}"
        
        # Verify evaluate was called for each content
        assert mock_judge.evaluate.call_count == 3

    def test_run_with_custom_evaluation_params(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test run method with custom evaluation parameters."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        custom_evaluation_params = {
            "criteria": "Evaluate the technical accuracy and clarity of the explanation",
            "input": "Explain {subject}",
            "rubric": {
                1.0: "Excellent technical accuracy and crystal clear explanation",
                0.8: "Good accuracy with minor clarity issues", 
                0.6: "Adequate but some technical errors or unclear sections",
                0.4: "Poor accuracy or very unclear",
                0.0: "Completely inaccurate or incomprehensible"
            },
            "scale": [0, 1],
            "context": "This is for a computer science course about {subject}",
            "template_vars": {"subject": "quantum computing"}
        }
        content = "Quantum computing uses qubits to perform calculations, which are quantum bits that can exist in multiple states simultaneously."
        request = ContentAnalysisHttpRequest(
            contents=[content],
            detector_params=custom_evaluation_params
        )
        
        result = asyncio.run(detector.analyze_content(request))
        
        # Verify complex parameters were passed correctly
        expected_call_params = {
            "content": content,
            **custom_evaluation_params
        }
        mock_judge.evaluate.assert_called_once_with(**expected_call_params)

    def test_list_available_metrics(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test listing available metrics."""
        detector: LLMJudgeDetector
        detector, _ = detector_with_mock_judge
        
        metrics = detector.list_available_metrics()
        
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        assert "safety" in metrics
        assert "helpfulness" in metrics
        assert "accuracy" in metrics
        # Verify it's sorted
        assert metrics == sorted(metrics)

    def test_close_detector(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test closing the detector properly closes the judge."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        asyncio.run(detector.close())
        
        mock_judge.close.assert_called_once()


class TestLLMJudgeDetectorGenerationAnalysis:
    """Test suite for LLMJudgeDetector generation analysis."""
    
    @pytest.fixture
    def mock_judge_result(self) -> EvaluationResult:
        """Mock EvaluationResult for generation testing."""
        return EvaluationResult(
            decision="HELPFUL",
            reasoning="This generated response is helpful and addresses the user's question appropriately.",
            score=0.85,
            metadata={"model": "test-model"}
        )
    
    @pytest.fixture
    def detector_with_mock_judge(self, mock_judge_result) -> Tuple[LLMJudgeDetector, AsyncMock]:
        """Create detector with mocked Judge."""
        with patch.dict(os.environ, {"VLLM_BASE_URL": "http://test:8000"}):
            with patch('vllm_judge.Judge.from_url') as mock_judge_class:
                # Create mock judge instance
                mock_judge_instance = AsyncMock()
                mock_judge_instance.evaluate = AsyncMock(return_value=mock_judge_result)
                mock_judge_instance.config.model = "test-model"
                mock_judge_instance.config.base_url = "http://test:8000"
                mock_judge_instance.close = AsyncMock()
                
                mock_judge_class.return_value = mock_judge_instance
                
                detector = LLMJudgeDetector()
                return detector, mock_judge_instance

    def test_evaluate_single_generation_basic_metric(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test basic generation evaluation with just a metric."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        prompt = "What is artificial intelligence?"
        generated_text = "Artificial intelligence (AI) refers to the simulation of human intelligence in machines."
        params = {"metric": "helpfulness"}
        
        result = asyncio.run(detector.evaluate_single_generation(prompt, generated_text, params))
        
        # Verify judge.evaluate was called correctly
        mock_judge.evaluate.assert_called_once_with(
            input=prompt,
            content=generated_text,
            metric="helpfulness"
        )
        
        # Verify response format
        assert isinstance(result, GenerationAnalysisResponse)
        assert result.detection == "HELPFUL"
        assert result.score == 0.85
        assert result.detection_type == "llm_judge" 
        assert "reasoning" in result.metadata

    def test_evaluate_single_generation_full_parameters(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test generation evaluation with all vLLM Judge parameters."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        prompt = "Explain quantum computing in simple terms"
        generated_text = "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, allowing for parallel processing of information."
        params = {
            "criteria": "accuracy, clarity, and completeness",
            "rubric": "Score based on technical accuracy and accessibility",
            "scale": [1, 10],
            "examples": [{"input": "test prompt", "output": "example response"}],
            "system_prompt": "You are evaluating educational content",
            "context": "This is for a general audience explanation of {topic}",
            "template_vars": {"topic": "quantum computing"}
        }
        
        asyncio.run(detector.evaluate_single_generation(prompt, generated_text, params))
        
        # Verify all parameters were passed through
        expected_call = {
            "input": prompt,
            "content": generated_text,
            **params
        }
        mock_judge.evaluate.assert_called_once_with(**expected_call)

    def test_evaluate_single_generation_criteria_without_metric(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test generation evaluation with criteria but no metric (should default scale)."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        prompt = "Write a short story"
        generated_text = "Once upon a time, there was a brave knight who saved a village from a dragon."
        params = {
            "criteria": "creativity and engagement",
            "rubric": "Custom rubric for story evaluation"
        }
        
        asyncio.run(detector.evaluate_single_generation(prompt, generated_text, params))
        
        # Should add default scale when criteria provided without metric
        expected_params = {
            "input": prompt,
            "content": generated_text,
            "criteria": "creativity and engagement",
            "rubric": "Custom rubric for story evaluation",
            "scale": (0, 1)
        }
        mock_judge.evaluate.assert_called_once_with(**expected_params)

    def test_evaluate_single_generation_no_params(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test generation evaluation with no parameters (should default to safety)."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        prompt = "Tell me about AI"
        generated_text = "AI is a field of computer science focused on creating intelligent machines."
        params = {}
        
        asyncio.run(detector.evaluate_single_generation(prompt, generated_text, params))
        
        # Should default to safety metric
        expected_params = {
            "input": prompt,
            "content": generated_text,
            "metric": "safety"
        }
        mock_judge.evaluate.assert_called_once_with(**expected_params)

    def test_evaluate_single_generation_invalid_metric(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test generation evaluation with invalid metric raises error."""
        detector: LLMJudgeDetector
        detector, _ = detector_with_mock_judge
        
        prompt = "Test prompt"
        generated_text = "Test generation"
        params = {"metric": "invalid_metric"}
        
        with pytest.raises(MetricNotFoundError, match="Metric 'invalid_metric' not found"):
            asyncio.run(detector.evaluate_single_generation(prompt, generated_text, params))

    def test_analyze_generation_basic_request(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test the analyze_generation method with basic request."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        request = GenerationAnalysisHttpRequest(
            prompt="What is machine learning?",
            generated_text="Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
            detector_params={"metric": "accuracy"}
        )
        
        result = asyncio.run(detector.analyze_generation(request))
        
        # Verify judge.evaluate was called correctly
        mock_judge.evaluate.assert_called_once_with(
            input="What is machine learning?",
            content="Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
            metric="accuracy"
        )
        
        # Verify response format
        assert isinstance(result, GenerationAnalysisResponse)
        assert result.detection == "HELPFUL"
        assert result.score == 0.85
        assert result.detection_type == "llm_judge"
        assert "reasoning" in result.metadata
        assert result.metadata["reasoning"] is not None

    def test_analyze_generation_complex_request(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test the analyze_generation method with complex parameters."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        request = GenerationAnalysisHttpRequest(
            prompt="Explain the benefits and risks of artificial intelligence",
            generated_text="AI offers significant benefits like improved efficiency and automation, but also poses risks such as job displacement and potential bias in decision-making systems.",
            detector_params={
                "criteria": "balance, accuracy, and completeness",
                "rubric": {
                    1.0: "Excellent balance of benefits and risks with high accuracy",
                    0.8: "Good coverage with minor gaps",
                    0.6: "Adequate but missing some key points",
                    0.4: "Poor coverage or significant inaccuracies",
                    0.0: "Completely inadequate or misleading"
                },
                "scale": [0, 1],
                "context": "This is for an educational discussion about AI ethics"
            }
        )
        
        result = asyncio.run(detector.analyze_generation(request))
        
        # Verify complex parameters were passed correctly
        expected_call_params = {
            "input": request.prompt,
            "content": request.generated_text,
            **request.detector_params
        }
        mock_judge.evaluate.assert_called_once_with(**expected_call_params)
        
        # Verify response
        assert isinstance(result, GenerationAnalysisResponse)
        assert result.detection_type == "llm_judge"

    def test_analyze_generation_empty_params(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test analyze_generation with empty detector params (should default to safety)."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        request = GenerationAnalysisHttpRequest(
            prompt="Hello, how are you?",
            generated_text="I'm doing well, thank you for asking! How can I assist you today?",
            detector_params={}
        )
        
        result = asyncio.run(detector.analyze_generation(request))
        
        # Should default to safety metric
        expected_params = {
            "input": request.prompt,
            "content": request.generated_text,
            "metric": "safety"
        }
        mock_judge.evaluate.assert_called_once_with(**expected_params)
        
        assert isinstance(result, GenerationAnalysisResponse)
        assert result.detection_type == "llm_judge"

    def test_generation_analysis_with_numeric_score(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test generation analysis handles numeric scores correctly."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        # Mock a numeric decision result
        numeric_result = EvaluationResult(
            decision=8.5,
            reasoning="High quality response with good accuracy",
            score=8.5,
            metadata={"model": "test-model"}
        )
        mock_judge.evaluate.return_value = numeric_result
        
        request = GenerationAnalysisHttpRequest(
            prompt="Explain photosynthesis",
            generated_text="Photosynthesis is the process by which plants convert light energy into chemical energy.",
            detector_params={"metric": "accuracy", "scale": [0, 10]}
        )
        
        result = asyncio.run(detector.analyze_generation(request))
        
        assert isinstance(result, GenerationAnalysisResponse)
        assert result.detection == "8.5"
        assert result.score == 8.5
        assert result.detection_type == "llm_judge"

    def test_generation_analysis_with_none_score(self, detector_with_mock_judge: Tuple[LLMJudgeDetector, AsyncMock]) -> None:
        """Test generation analysis handles None scores correctly."""
        detector: LLMJudgeDetector
        mock_judge: AsyncMock
        detector, mock_judge = detector_with_mock_judge
        
        # Mock a result with None score
        none_score_result = EvaluationResult(
            decision="GOOD",
            reasoning="Good quality response",
            score=None,
            metadata={"model": "test-model"}
        )
        mock_judge.evaluate.return_value = none_score_result
        
        request = GenerationAnalysisHttpRequest(
            prompt="Test prompt",
            generated_text="Test generation",
            detector_params={"metric": "helpfulness"}
        )
        
        result = asyncio.run(detector.analyze_generation(request))
        
        assert isinstance(result, GenerationAnalysisResponse)
        assert result.detection == "GOOD"
        assert result.score == 0.0  # Should default to 0.0 when score is None
        assert result.detection_type == "llm_judge"