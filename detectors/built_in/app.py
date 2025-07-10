from fastapi import HTTPException

from base_detector_registry import BaseDetectorRegistry
from regex_detectors import RegexDetectorRegistry
from file_type_detectors import FileTypeDetectorRegistry

from prometheus_fastapi_instrumentator import Instrumentator
from detectors.common.scheme import ContentAnalysisHttpRequest,  ContentsAnalysisResponse
from detectors.common.app import DetectorBaseAPI as FastAPI

app = FastAPI()
Instrumentator().instrument(app).expose(app)


registry : dict[str, BaseDetectorRegistry] = {
    "regex": RegexDetectorRegistry(),
    "file_type": FileTypeDetectorRegistry(),
}

@app.post("/api/v1/text/contents", response_model=ContentsAnalysisResponse)
def detect_content(request: ContentAnalysisHttpRequest):
    detections = []
    for content in request.contents:
        message_detections = []
        for detector_kind, detector_registry in registry.items():
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
    for detector_type, detector_registry in registry.items():
        result[detector_type] = {}
        for detector_name, detector_fn in detector_registry.get_registry().items():
            result[detector_type][detector_name] = detector_fn.__doc__
    return result