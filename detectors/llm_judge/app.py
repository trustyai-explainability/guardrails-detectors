from contextlib import asynccontextmanager
from typing import Annotated, Dict

from fastapi import Header, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from guardrails_detector_common import DetectorBaseAPI as FastAPI
from detectors.llm_judge.detector import LLMJudgeDetector
from detectors.common.scheme import (
    ContentAnalysisHttpRequest,
    ContentsAnalysisResponse,
    MetricsListResponse,
    Error,
    GenerationAnalysisHttpRequest,
    GenerationAnalysisResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""

    app.set_detector(LLMJudgeDetector())
    yield
    # Clean up resources
    detector: LLMJudgeDetector = app.get_detector()
    if detector and hasattr(detector, 'close'):
        await detector.close()
    app.cleanup_detector()


app = FastAPI(lifespan=lifespan, dependencies=[])
Instrumentator().instrument(app).expose(app)


@app.post(
    "/api/v1/text/contents",
    response_model=ContentsAnalysisResponse,
    description="""LLM-as-Judge detector that evaluates content using various metrics like safety, toxicity, accuracy, helpfulness, etc. \
                    The metric detector_params parameter allows you to specify which evaluation criteria to use. \
                    Supports all built-in vllm_judge metrics including safety, accuracy, helpfulness, clarity, and many more.""",
    responses={
        404: {"model": Error, "description": "Resource Not Found"},
        422: {"model": Error, "description": "Validation Error"},
    },
)
async def detector_content_analysis_handler(
    request: ContentAnalysisHttpRequest,
    detector_id: Annotated[str, Header(example="llm_judge_safety")],
):
    """Analyze content using LLM-as-Judge evaluation."""
    detector: LLMJudgeDetector = app.get_detector()
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not found")
    return ContentsAnalysisResponse(root=await detector.analyze_content(request))

@app.post(
    "/api/v1/text/generation",
    response_model=GenerationAnalysisResponse,
    description="""Analyze a single generation using the specified metric. \
                    The metric detector_params parameter allows you to specify which evaluation criteria to use. \
                    Supports all built-in vllm_judge metrics including safety, accuracy, helpfulness, clarity, and many more.""",
    responses={
        404: {"model": Error, "description": "Resource Not Found"},
        422: {"model": Error, "description": "Validation Error"},
    },
)
async def detector_generation_analysis_handler(
    request: GenerationAnalysisHttpRequest,
    detector_id: Annotated[str, Header(example="llm_judge_safety")],
):
    """Analyze a single generation using LLM-as-Judge evaluation."""
    detector: LLMJudgeDetector = app.get_detector()
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not found")
    return await detector.analyze_generation(request)

@app.get(
    "/api/v1/metrics",
    response_model=MetricsListResponse,
    description="List all available metrics for LLM Judge evaluation",
    responses={
        404: {"model": Error, "description": "Resource Not Found"},
    },
)
async def list_metrics():
    """List all available evaluation metrics."""
    detector: LLMJudgeDetector = app.get_detector()
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not found")
    
    metrics = detector.list_available_metrics()
    return MetricsListResponse(metrics=metrics, total=len(metrics))