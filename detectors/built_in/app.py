import json
import logging

from fastapi import HTTPException, Request
from contextlib import asynccontextmanager
from base_detector_registry import BaseDetectorRegistry
from regex_detectors import RegexDetectorRegistry
from custom_detectors_wrapper import CustomDetectorRegistry
from file_type_detectors import FileTypeDetectorRegistry

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, multiprocess
from starlette.responses import Response
from detectors.common.scheme import ContentAnalysisHttpRequest,  ContentsAnalysisResponse
from detectors.common.app import DetectorBaseAPI as FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    for detector_registry in [
        RegexDetectorRegistry(),
        FileTypeDetectorRegistry(),
        CustomDetectorRegistry()
    ]:
        app.set_detector(detector_registry, detector_registry.registry_name)
        detector_registry.set_instruments(app.state.instruments)
    yield
    app.cleanup_detector()


app = FastAPI(lifespan=lifespan)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/metrics")
def metrics():
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    data = generate_latest(registry)
    return Response(data, media_type=CONTENT_TYPE_LATEST)

@app.post("/api/v1/text/contents", response_model=ContentsAnalysisResponse)
def detect_content(request: ContentAnalysisHttpRequest, raw_request: Request):
    logger.info(f"Request for {request.detector_params}")

    headers = dict(raw_request.headers)

    detections = []
    for content in request.contents:
        message_detections = []
        for detector_kind in request.detector_params:
            detector_registry = app.get_all_detectors().get(detector_kind)
            if detector_registry is None:
                raise HTTPException(status_code=400, detail=f"Detector {detector_kind} not found")
            if not isinstance(detector_registry, BaseDetectorRegistry):
                raise TypeError(f"Detector {detector_kind} is not a valid BaseDetectorRegistry")
            else:
                try:
                    message_detections += detector_registry.handle_request(content, request.detector_params, headers)
                except HTTPException as e:
                    raise e
                except Exception as e:
                    raise HTTPException(status_code=500) from e
        detections.append(message_detections)
    return ContentsAnalysisResponse(root=detections)


@app.get("/registry")
def get_registry():
    result = {}
    for detector_type, detector_registry in app.get_all_detectors().items():
        if not isinstance(detector_registry, BaseDetectorRegistry):
            raise TypeError(f"Detector {detector_type} is not a valid BaseDetectorRegistry")
        result[detector_type] = {}
        for detector_name, detector_fn in detector_registry.get_registry().items():
            docstring = detector_fn.__doc__
            try:
                # Try to parse as JSON
                parsed = json.loads(docstring)
                result[detector_type][detector_name] = parsed
            except Exception:
                result[detector_type][detector_name] = docstring
    return result