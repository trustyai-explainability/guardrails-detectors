from fastapi import HTTPException
from contextlib import asynccontextmanager
from base_detector_registry import BaseDetectorRegistry
from regex_detectors import RegexDetectorRegistry
from file_type_detectors import FileTypeDetectorRegistry

from prometheus_fastapi_instrumentator import Instrumentator
from detectors.common.scheme import ContentAnalysisHttpRequest,  ContentsAnalysisResponse
from detectors.common.app import DetectorBaseAPI as FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.set_detector(RegexDetectorRegistry(), "regex")
    app.set_detector(FileTypeDetectorRegistry(), "file_type")
    yield
    
    app.cleanup_detector()


app = FastAPI(lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


# registry : dict[str, BaseDetectorRegistry] = {
#     "regex": RegexDetectorRegistry(),
#     "file_type": FileTypeDetectorRegistry(),
# }

@app.post("/api/v1/text/contents", response_model=ContentsAnalysisResponse)
def detect_content(request: ContentAnalysisHttpRequest):
    detections = []
    for content in request.contents:
        message_detections = []
        for detector_kind, detector_registry in app.get_all_detectors().items():
            assert isinstance(detector_registry, BaseDetectorRegistry), f"Detector {detector_kind} is not a valid BaseDetectorRegistry"
            if detector_kind in request.detector_params:
                try:
                    message_detections += detector_registry.handle_request(content, request.detector_params)
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
        assert isinstance(detector_registry, BaseDetectorRegistry), f"Detector {detector_type} is not a valid BaseDetectorRegistry"
        result[detector_type] = {}
        for detector_name, detector_fn in detector_registry.get_registry().items():
            result[detector_type][detector_name] = detector_fn.__doc__
    return result