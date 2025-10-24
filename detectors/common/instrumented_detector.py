import contextlib
import time


class InstrumentedDetector:
    def __init__(self, registry_name: str = "default"):
        self.registry_name = registry_name
        self.instruments = {}

    @contextlib.contextmanager
    def instrument_runtime(self, function_name: str):
        try:
            start_time = time.time()
            yield
            if self.instruments.get("runtime"):
                self.instruments["runtime"].labels(self.registry_name, function_name).inc(time.time() - start_time)
        finally:
            pass

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
