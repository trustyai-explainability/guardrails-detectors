import os
import sys
from contextlib import asynccontextmanager
from typing import Annotated, Dict

from fastapi import Header
from prometheus_fastapi_instrumentator import Instrumentator
sys.path.insert(0, os.path.abspath(".."))

from common.app import DetectorBaseAPI as FastAPI
from .detector import LLMJudgeDetector
from .scheme import (
    ContentAnalysisHttpRequest,
    ContentsAnalysisResponse,
    MetricsListResponse,
    Error,
)

detector_objects: Dict[str, LLMJudgeDetector] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    try:
        detector_objects["detector"] = LLMJudgeDetector()
        yield
    finally:
        # Clean up resources
        if "detector" in detector_objects:
            await detector_objects["detector"].close()
        detector_objects.clear()


app = FastAPI(lifespan=lifespan, dependencies=[])
Instrumentator().instrument(app).expose(app)


@app.post(
    "/api/v1/text/contents",
    response_model=ContentsAnalysisResponse,
    description="""LLM-as-Judge detector that evaluates content using various metrics like safety, toxicity, accuracy, helpfulness, etc. \
                    The metric parameter allows you to specify which evaluation criteria to use. \
                    Supports all built-in vllm_judge metrics including safety, accuracy, helpfulness, clarity, and many more.""",
    responses={
        404: {"model": Error, "description": "Resource Not Found"},
        422: {"model": Error, "description": "Validation Error"},
    },
)
async def detector_unary_handler(
    request: ContentAnalysisHttpRequest,
    detector_id: Annotated[str, Header(example="llm_judge_safety")],
):
    """Analyze content using LLM-as-Judge evaluation."""
    return ContentsAnalysisResponse(root=await detector_objects["detector"].run(request))


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
    detector = detector_objects.get("detector")
    if not detector:
        return {"metrics": [], "total": 0}
    
    metrics = detector.list_available_metrics()
    return MetricsListResponse(metrics=metrics, total=len(metrics))