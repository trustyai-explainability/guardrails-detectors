import logging

from fastapi import HTTPException, Request
from contextlib import asynccontextmanager
from base_detector_registry import BaseDetectorRegistry
from regex_detectors import RegexDetectorRegistry
from custom_detectors_wrapper import CustomDetectorRegistry
from file_type_detectors import FileTypeDetectorRegistry

from prometheus_fastapi_instrumentator import Instrumentator
from detectors.common.scheme import ContentAnalysisHttpRequest,  ContentsAnalysisResponse
from detectors.common.app import DetectorBaseAPI as FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.set_detector(RegexDetectorRegistry(), "regex")
    app.set_detector(FileTypeDetectorRegistry(), "file_type")
    app.set_detector(CustomDetectorRegistry(), "custom")
    yield
    
    app.cleanup_detector()


app = FastAPI(lifespan=lifespan)
Instrumentator().instrument(app).expose(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            result[detector_type][detector_name] = detector_fn.__doc__
    return result