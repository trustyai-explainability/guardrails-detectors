import os
import sys
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Header
from prometheus_fastapi_instrumentator import Instrumentator

sys.path.insert(0, os.path.abspath(".."))

from detector import Detector

from detectors.common.app import DetectorBaseAPI as FastAPI
from detectors.common.scheme import (
    ContentAnalysisHttpRequest,
    ContentsAnalysisResponse,
    Error,
)

detector_objects = {}


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
    return ContentsAnalysisResponse(root=detector_objects["detector"].run(request))