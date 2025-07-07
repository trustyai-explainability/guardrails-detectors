import os
from typing import List, Dict, Any

from vllm_judge import Judge, EvaluationResult, BUILTIN_METRICS
from vllm_judge.exceptions import MetricNotFoundError
from detectors.common.app import logger
from detectors.llm_judge.scheme import (
    ContentAnalysisHttpRequest,
    ContentAnalysisResponse,
    ContentsAnalysisResponse,
    GenerationAnalysisHttpRequest,
    GenerationAnalysisResponse,
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
    
    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make sure the params have valid metric/criteria and scale.
        """
        if "metric" not in params:
            if "criteria" not in params:
                params["metric"] = "safety" # Default to safety
            elif "scale" not in params:
                params["scale"] = (0, 1) # Default to 0-1 scale
        else:
            if params["metric"] not in self.available_metrics:
                raise MetricNotFoundError(
                    f"Metric '{params['metric']}' not found. Available metrics: {', '.join(sorted(self.available_metrics))}"
                )
            judge_metric = BUILTIN_METRICS[params["metric"]]
            if judge_metric.scale is None:
                params["scale"] = (0, 1) # Default to 0-1 scale
        
        return params
    
    def _get_score(self, result: EvaluationResult) -> float:
        """
        Get the score from the evaluation result.
        """
        if isinstance(result.decision, (int, float)) or result.score is not None:
            return float(result.score if result.score is not None else result.decision)
        return 0.0 # FIXME: default to 0 because of non-optional field in schema

    async def evaluate_single_content(self, content: str, params: Dict[str, Any]) -> ContentAnalysisResponse:
        """
        Evaluate a single piece of content using the specified metric.
        
        Args:
            content: Text content to evaluate
            params: vLLM Judge parameters for the evaluation
            
        Returns:
            ContentAnalysisResponse with evaluation results
        """
        params: Dict[str, Any] = self._validate_params(params)

        evaluation_params: Dict[str, Any] = {
            "content": content,
            **params
        }
        
        # Perform evaluation
        result: EvaluationResult = await self.judge.evaluate(
            **evaluation_params
        )
        
        # Convert to response format. 
        score: float = self._get_score(result)
        
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

    async def analyze_content(self, request: ContentAnalysisHttpRequest) -> ContentsAnalysisResponse:
        """
        Run content analysis for each input text.
        
        Args:
            request: Input request containing texts and optional metric to analyze
            
        Returns:
            ContentsAnalysisResponse: The aggregated response for all input texts
        """

        contents_analyses = []
        
        for content in request.contents:
            analysis = await self.evaluate_single_content(content, request.detector_params)
            contents_analyses.append([analysis])  # Wrap in list to match schema
        
        return contents_analyses

    async def evaluate_single_generation(self, prompt: str, generated_text: str, params: Dict[str, Any]) -> GenerationAnalysisResponse:
        """
        Evaluate a single generation based on the prompt and generated text.

        Args:
            prompt: Prompt to the LLM
            generated_text: Generated text from the LLM
            params: vLLM Judge parameters for the evaluation
            
        Returns:
            GenerationAnalysisResponse: The response for the generation analysis
        """
        params: Dict[str, Any] = self._validate_params(params)
        evaluation_params: Dict[str, Any] = {
            "input": prompt,
            "content": generated_text,
            **params
        }

        result: EvaluationResult = await self.judge.evaluate(
            **evaluation_params
        )
        
        score: float = self._get_score(result)
        
        return GenerationAnalysisResponse(
            detection=str(result.decision),
            detection_type="llm_judge",
            score=score,
            evidences=[],
            metadata={"reasoning": result.reasoning}
        )

    async def analyze_generation(self, request: GenerationAnalysisHttpRequest) -> GenerationAnalysisResponse:
        """
        Analyze a single generation based on the prompt and generated text.

        Args:
            request: Input request containing prompt, generated text and optional metric to analyze
            
        Returns:
            GenerationAnalysisResponse: The response for the generation analysis
        """
        return await self.evaluate_single_generation(prompt=request.prompt,
                                                     generated_text=request.generated_text,
                                                     params=request.detector_params)
    
    async def close(self):
        """Close the judge client."""
        if self.judge:
            await self.judge.close()
    
    def list_available_metrics(self) -> List[str]:
        """Return list of available metrics."""
        return sorted(list(self.available_metrics))