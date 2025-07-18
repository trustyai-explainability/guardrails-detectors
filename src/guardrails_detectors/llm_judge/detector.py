import os
from typing import List, Dict, Any

from vllm_judge import Judge, EvaluationResult, BUILTIN_METRICS
from vllm_judge.exceptions import MetricNotFoundError
from detectors.common.app import logger
from detectors.llm_judge.scheme import (
    ContentAnalysisHttpRequest,
    ContentAnalysisResponse,
    ContentsAnalysisResponse,
)


class LLMJudgeDetector:
    """LLM-as-Judge detector for evaluating content using vllm_judge."""
    
    def __init__(self) -> None:
        """Initialize the LLM Judge Detector."""
        self.judge: Judge = None
        self.available_metrics = set(BUILTIN_METRICS.keys())
        
        # Get configuration from environment
        self.vllm_base_url = os.environ.get("VLLM_BASE_URL")
        
        if not self.vllm_base_url:
            raise ValueError("VLLM_BASE_URL environment variable is required")
        
        logger.info(f"Initializing LLM Judge with URL: {self.vllm_base_url}")
        
        self._initialize_judge()
    
    def _initialize_judge(self) -> None:
        """Initialize the vLLM Judge."""
        try:
            self.judge = Judge.from_url(base_url=self.vllm_base_url)
            logger.info(f"LLM Judge initialized successfully with model: {self.judge.config.model} and base url: {self.judge.config.base_url}")
            logger.info(f"Available metrics: {', '.join(sorted(self.available_metrics))}")
            
        except Exception as e:
            logger.error(f"Failed to detect model: {e}")
            raise
    
    async def evaluate_single_content(self, content: str, params: Dict[str, Any]) -> ContentAnalysisResponse:
        """
        Evaluate a single piece of content using the specified metric.
        
        Args:
            content: Text content to evaluate
            params: vLLM Judge parameters for the evaluation
            
        Returns:
            ContentAnalysisResponse with evaluation results
        """
        if "metric" not in params:
            if "criteria" not in params:
                params["metric"] = "safety" # Default to safety
            elif "scale" not in params:
                params["scale"] = (0, 1) # Default to 0-1 scale
        
        if "metric" in params:
            if params["metric"] not in self.available_metrics:
                raise MetricNotFoundError(
                    f"Metric '{params['metric']}' not found. Available metrics: {', '.join(sorted(self.available_metrics))}"
                )
            judge_metric = BUILTIN_METRICS[params["metric"]]
            if judge_metric.scale is None:
                params["scale"] = (0, 1) # Default to 0-1 scale

        evaluation_params = {
            "content": content,
            **params
        }
        
        # Perform evaluation
        result: EvaluationResult = await self.judge.evaluate(
            **evaluation_params
        )
        
        # Convert to response format
        score = None
        if isinstance(result.decision, (int, float)) or result.score is not None:
            # Numeric result
            score = float(result.score if result.score is not None else result.decision)
        
        return ContentAnalysisResponse(
            start=0,
            end=len(content),
            detection=str(result.decision),
            detection_type="llm_judge",
            score=score,
            text=content,
            evidences=[],
            metadata={"reasoning": result.reasoning}
        )

    async def run(self, request: ContentAnalysisHttpRequest) -> ContentsAnalysisResponse:
        """
        Run content analysis for each input text.
        
        Args:
            request: Input request containing texts and metric to analyze
            
        Returns:
            ContentsAnalysisResponse: The aggregated response for all input texts
        """

        contents_analyses = []
        
        for content in request.contents:
            analysis = await self.evaluate_single_content(content, request.detector_params)
            contents_analyses.append([analysis])  # Wrap in list to match schema
        
        return contents_analyses
            
    
    async def close(self):
        """Close the judge client."""
        if self.judge:
            await self.judge.close()
    
    def list_available_metrics(self) -> List[str]:
        """Return list of available metrics."""
        return sorted(list(self.available_metrics))