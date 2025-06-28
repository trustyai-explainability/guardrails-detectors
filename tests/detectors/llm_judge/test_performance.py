import pytest
import time
import asyncio
import os
from unittest.mock import patch, AsyncMock

from detectors.llm_judge.detector import LLMJudgeDetector
from detectors.llm_judge.scheme import ContentAnalysisHttpRequest
from vllm_judge import EvaluationResult


class TestPerformance:
    """Performance and concurrency tests."""
    
    def test_concurrent_evaluations(self) -> None:
        """Test handling multiple concurrent evaluations."""
        with patch.dict(os.environ, {"VLLM_BASE_URL": "http://test:8000"}):
            with patch('vllm_judge.Judge.from_url') as mock_judge_class:
                # Mock judge with slight delay to simulate real processing
                mock_judge = AsyncMock()
                async def mock_evaluate(**kwargs):
                    await asyncio.sleep(0.1)  # Simulate processing time
                    return EvaluationResult(
                        decision="SAFE",
                        reasoning="Safe content",
                        score=0.9,
                        metadata={}
                    )
                mock_judge.evaluate = mock_evaluate
                mock_judge.config.model = "test-model"
                mock_judge.config.base_url = "http://test:8000"
                mock_judge_class.return_value = mock_judge
                
                detector = LLMJudgeDetector()
                
                # Test concurrent processing
                contents = [f"Test content {i}" for i in range(10)]
                async def run_concurrent_evaluations():
                    start_time = time.time()
                    
                    tasks = []
                    for content in contents:
                        task = detector.evaluate_single_content(
                            content, {"metric": "safety"}
                        )
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks)
                    end_time = time.time()
                    
                    return results, end_time - start_time
            
                results, duration = asyncio.run(run_concurrent_evaluations())
                
                # Should complete in roughly 0.1 seconds (concurrent) rather than 1 second (sequential)
                assert duration < 0.5
                assert len(results) == 10
                
                for i, result in enumerate(results):
                    assert result.text == f"Test content {i}"
    
    def test_batch_processing_performance(self) -> None:
        """Test performance of batch processing."""
        with patch.dict(os.environ, {"VLLM_BASE_URL": "http://test:8000"}):
            with patch('vllm_judge.Judge.from_url') as mock_judge_class:
                mock_judge = AsyncMock()
                mock_judge.evaluate = AsyncMock(return_value=EvaluationResult(
                    decision="SAFE", reasoning="Safe", score=0.9, metadata={}
                ))
                mock_judge.config.model = "test-model"
                mock_judge.config.base_url = "http://test:8000"
                mock_judge_class.return_value = mock_judge
                
                detector = LLMJudgeDetector()
                
                # Large batch request
                request = ContentAnalysisHttpRequest(
                    contents=[f"Content {i}" for i in range(100)],
                    detector_params={"metric": "safety"}
                )
                
                start_time = time.time()
                result = asyncio.run(detector.run(request))
                end_time = time.time()
                
                assert len(result) == 100
                assert end_time - start_time < 0.5 # Should be reasonably fast