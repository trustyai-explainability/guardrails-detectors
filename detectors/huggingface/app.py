from contextlib import asynccontextmanager
from typing import List

from prometheus_fastapi_instrumentator import Instrumentator
from starlette.concurrency import run_in_threadpool
from detectors.common.app import DetectorBaseAPI as FastAPI
from detectors.huggingface.detector import Detector
from detectors.common.scheme import (
    ContentAnalysisHttpRequest,
    ContentsAnalysisResponse,
    Error,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector = Detector()
    app.set_detector(detector, detector.model_name)
    detector.add_instruments(app.state.instruments)
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
):
    detectors: List[Detector] = list(app.get_all_detectors().values())
    if not len(detectors) or not detectors[0]:
        raise RuntimeError("Detector is not initialized")
    result = await run_in_threadpool(detectors[0].run, request)
    return ContentsAnalysisResponse(root=result)

