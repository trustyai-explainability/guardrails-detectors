import contextlib
import logging
from abc import ABC, abstractmethod
import time
from http.client import HTTPException
from typing import List

from detectors.common.scheme import ContentAnalysisResponse
from detectors.common.metrics import update_detection_rates

class BaseDetectorRegistry(ABC):
    def __init__(self, registry_name):
        self.registry = None
        self.registry_name = registry_name

        # prometheus
        self.instruments = None
        self.detections_gauge = None

    @abstractmethod
    def handle_request(self, content: str, detector_params: dict, headers: dict, **kwargs) -> List[ContentAnalysisResponse]:
        pass
    
    def get_registry(self):
        return self.registry

    def add_instruments(self, gauges):
        self.instruments = gauges

    def increment_detector_instruments(self, function_name: str, is_detection: bool):
        """Increment the detection and request counters, automatically update rates"""
        updated = False
        if self.instruments.get("requests"):
            self.instruments["requests"].labels(self.registry_name, function_name).inc()
            updated = True
        if is_detection and self.instruments.get("detections"):
            self.instruments["detections"].labels(self.registry_name, function_name).inc()
            updated = True

        if updated:
            update_detection_rates(self.instruments, self.registry_name, function_name)

    def increment_error_instruments(self, function_name: str):
        """Increment the error counter, update the rate gauges"""
        if self.instruments.get("errors"):
            self.instruments["errors"].labels(self.registry_name, function_name).inc()
            update_detection_rates(self.instruments, self.registry_name, function_name)


    def throw_internal_detector_error(self, function_name: str, logger: logging.Logger, exception: Exception, increment_requests: bool):
        """consistent handling of internal errors within a detection function"""
        if increment_requests and self.instruments.get("requests"):
            self.instruments["requests"].labels(self.registry_name, function_name).inc()
        self.increment_error_instruments(function_name)
        logger.error(exception)
        raise HTTPException(status_code=500, detail="Detection error, check detector logs")


    @contextlib.contextmanager
    def instrument_runtime(self, function_name: str):
        try:
            start_time = time.time()
            yield
            self.instruments["runtime"].labels(self.registry_name, function_name).inc(time.time() - start_time)
        finally:
            pass

    def get_detection_functions_from_params(self, params: dict):
        """Parse the request parameters to extract and normalize detection functions as iterable list"""
        if self.registry_name in params and isinstance(params[self.registry_name], (list, str)):
            funcs = params[self.registry_name]
            return [funcs] if isinstance(funcs, str) else funcs
        else:
            return []