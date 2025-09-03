from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Header
from prometheus_fastapi_instrumentator import Instrumentator

from guardrails_detector_common import DetectorBaseAPI as FastAPI
from .detector import Detector
from  guardrails_detectors_common.scheme import (
    ContentAnalysisHttpRequest,
    ContentsAnalysisResponse,
    Error,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.set_detector(Detector())
    yield
    # Clean up the ML models and release the resources
    detector: Detector = app.get_detector()
    if detector and hasattr(detector, 'close'):
        detector.close()
    app.cleanup_detector()


app = FastAPI(lifespan=lifespan, dependencies=[])
Instrumentator().instrument(app).expose(app)


@app.post(
    "/api/v1/text/contents",
    response_model=ContentsAnalysisResponse,
    description="""Detectors that work on content text, be it user prompt or generated text. \
                    Generally classification type detectors qualify for this. <br>""",
    responses={
        404: {"model": Error, "description": "Resource Not Found"},
        422: {"model": Error, "description": "Validation Error"},
    },
)
async def detector_unary_handler(
    request: ContentAnalysisHttpRequest,
    detector_id: Annotated[str, Header(example="en_syntax_slate.38m.hap")],
):
    detector: Detector = app.get_detector()
    if not detector:
        raise RuntimeError("Detector is not initialized")
    return ContentsAnalysisResponse(root=detector.run(request))
