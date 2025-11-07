import logging
from abc import ABC, abstractmethod
from fastapi import HTTPException
from typing import List

from detectors.common.instrumented_detector import InstrumentedDetector
from detectors.common.scheme import ContentAnalysisResponse

class BaseDetectorRegistry(InstrumentedDetector, ABC):
    def __init__(self, registry_name):
        super().__init__(registry_name)
        self.registry = None


    @abstractmethod
    def handle_request(self, content: str, detector_params: dict, headers: dict, **kwargs) -> List[ContentAnalysisResponse]:
        pass
    
    def get_registry(self):
        return self.registry

    def throw_internal_detector_error(self, function_name: str, logger: logging.Logger, exception: Exception, increment_requests: bool):
        """consistent handling of internal errors within a detection function"""
        if increment_requests and self.instruments.get("requests"):
            self.instruments["requests"].labels(self.registry_name, function_name).inc()
        self.increment_error_instruments(function_name)
        logger.error(exception)
        raise HTTPException(status_code=500, detail="Detection error, check detector logs")


    def get_detection_functions_from_params(self, params: dict):
        """Parse the request parameters to extract and normalize detection functions as iterable list"""
        if self.registry_name in params and isinstance(params[self.registry_name], (list, str)):
            funcs = params[self.registry_name]
            return [funcs] if isinstance(funcs, str) else funcs
        else:
            return []