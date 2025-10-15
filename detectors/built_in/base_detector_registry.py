import contextlib
import logging
from abc import ABC, abstractmethod
import time
from http.client import HTTPException
from typing import List

from detectors.common.scheme import ContentAnalysisResponse

class BaseDetectorRegistry(ABC):
    def __init__(self, registry_name):
        self.registry = None
        self.registry_name = registry_name

        # prometheus
        self.instruments = {}

    @abstractmethod
    def handle_request(self, content: str, detector_params: dict, headers: dict, **kwargs) -> List[ContentAnalysisResponse]:
        pass
    
    def get_registry(self):
        return self.registry

    def add_instruments(self, gauges):
        self.instruments = gauges

    def increment_detector_instruments(self, function_name: str, is_detection: bool):
        """Increment the detection and request counters, automatically update rates"""
        if self.instruments.get("requests"):
            self.instruments["requests"].labels(self.registry_name, function_name).inc()

        # The labels() function will initialize the counters if not already created.
        # This prevents the counters not existing until they are first incremented
        # If the counters have already been created, this is just a cheap dict.get() call
        if self.instruments.get("errors"):
            _ = self.instruments["errors"].labels(self.registry_name, function_name)
        if self.instruments.get("runtime"):
            _ = self.instruments["runtime"].labels(self.registry_name, function_name)

        # create and/or increment the detection counter
        if self.instruments.get("detections"):
            detection_counter = self.instruments["detections"].labels(self.registry_name, function_name)
            if is_detection:
                detection_counter.inc()


    def increment_error_instruments(self, function_name: str):
        """Increment the error counter, update the rate gauges"""
        if self.instruments.get("errors"):
            self.instruments["errors"].labels(self.registry_name, function_name).inc()


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
            if self.instruments.get("runtime"):
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