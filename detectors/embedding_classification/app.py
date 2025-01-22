import os
import sys
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Header
sys.path.insert(0, os.path.abspath(".."))

from common.app import DetectorBaseAPI as FastAPI
from detector import Detector
from scheme import (
    ContentAnalysisHttpRequest,
    ContentsAnalysisResponse,
    Error,
)

detector_objects = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    detector_objects["detector"] = Detector()
    yield
    # Clean up the ML models and release the resources
    detector_objects.clear()


app = FastAPI(lifespan=lifespan, dependencies=[])


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
