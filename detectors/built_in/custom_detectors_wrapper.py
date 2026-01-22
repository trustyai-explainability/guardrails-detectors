import ast
import logging
import importlib.util
import inspect
import functools
import os
import sys
import traceback

from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException
from typing import List, Optional, Callable

from base_detector_registry import BaseDetectorRegistry
from detectors.common.app import METRIC_PREFIX
from detectors.common.scheme import ContentAnalysisResponse

logger = logging.getLogger(__name__)

def use_instruments(instruments: List):
    """Use this decorator to register the provided Prometheus instruments with the main /metrics endpoint"""
    def inner_layer_1(func):
        @functools.wraps(func)
        def inner_layer_2(*args, **kwargs):
            return func(*args, **kwargs)

        # check to see if "func" is already decorated, and only add the prometheus instruments field into the original function
        target = get_underlying_function(func)
        setattr(target, "prometheus_instruments", instruments)
        return inner_layer_2
    return inner_layer_1

def non_blocking(return_value):
    """
    Use this decorator to run the guardrail as a non-blocking background thread.

    The `return_value` is returned instantly to the caller of the /api/v1/text/contents, while
    the logic inside the function will run asynchronously in the background.
    """
    def inner_layer_1(func):
        @functools.wraps(func)
        def inner_layer_2(*args, **kwargs):
            executor = getattr(non_blocking, "_executor", None)
            if executor is None:
                executor = ThreadPoolExecutor()
                non_blocking._executor = executor
            def runner():
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Exception in non-blocking guardrail {func.__name__}: {e}")
            executor.submit(runner)

            # check to see if "func" is already decorated by `use_instruments`, and grab the instruments if so
            target = get_underlying_function(func)
            if hasattr(target, "prometheus_instruments"):
                setattr(target, "prometheus_instruments", target.prometheus_instruments)
            return return_value
        return inner_layer_2
    return inner_layer_1

forbidden_names = [use_instruments.__name__, non_blocking.__name__]

def get_underlying_function(func):
    if hasattr(func, "__wrapped__"):
        return get_underlying_function(func.__wrapped__)
    return func


def custom_func_wrapper(func: Callable, func_name: str, s: str, headers: dict, func_kwargs: dict=None) -> Optional[ContentAnalysisResponse]:
    """Convert a some f(text)->bool into a Detector response"""
    sig = inspect.signature(func)
    try:
        if headers is not None:
            if func_kwargs is None:
                result = func(s, headers=headers)
            else:
                result = func(s, headers=headers, **func_kwargs)
        else:
            if func_kwargs is None:
                result = func(s)
            else:
                result = func(s, **func_kwargs)

    except Exception as e:
        logging.error(f"Error when computing custom detector function {func_name}: {e}")
        raise e
    if result:
        if isinstance(result, bool):
            return ContentAnalysisResponse(
                start=0,
                end=len(s),
                text=s,
                detection_type=func_name,
                detection=func_name,
                score=1.0)
        elif isinstance(result, dict):
            try:
                return ContentAnalysisResponse(**result)
            except Exception as e:
                logging.error(f"Error when trying to build ContentAnalysisResponse from {func_name} response: {e}")
                raise e
        else:
            msg = f"Unsupported result type for custom detector function {func_name}, must be bool or ContentAnalysisResponse, got: {type(result)}"
            logging.error(msg)
            raise TypeError(msg)
    else:
        return None


def static_code_analysis(module_path, forbidden_imports=None, forbidden_calls=None):
    """
    Perform static code analysis on a Python module to check for forbidden imports and function calls.
    Returns a list of issues found.
    """
    if forbidden_imports is None:
        forbidden_imports = {"os", "subprocess", "sys", "shutil"}
    if forbidden_calls is None:
        forbidden_calls = {"eval", "exec", "open", "compile", "input"}

    issues = []
    with open(module_path, "r") as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=module_path)
    except Exception as e:
        issues.append(f"Failed to parse {module_path}: {e}")
        return issues

    for node in ast.walk(tree):
        # Check for forbidden imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in forbidden_imports:
                    issues.append(f"- Forbidden import: {alias.name} (line {node.lineno})")
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in forbidden_imports:
                # Allow specific exception: from os import environ
                if node.module == "os" and len(node.names) == 1 and node.names[0].name in {"environ", "getenv"}:
                    continue
                issues.append(f"- Forbidden import: {node.module} (line {node.lineno})")

        # Check for forbidden function calls
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = f"{getattr(node.func.value, 'id', '')}.{node.func.attr}"
            if func_name in forbidden_calls:
                issues.append(f"- Forbidden function call: {func_name} (line {node.lineno})")
    return issues


class CustomDetectorRegistry(BaseDetectorRegistry):
    def __init__(self):
        super().__init__("custom")

        # check the imported code for potential security issues
        issues = static_code_analysis(module_path = os.path.join(os.path.dirname(__file__), "custom_detectors", "custom_detectors.py"))
        if issues:
            logging.error(f"Detected {len(issues)} potential security issues inside the custom_detectors file: {issues}")
            raise ImportError(f"Unsafe code detected in custom_detectors:\n" + "\n".join(issues))

        # grab custom detectors module
        module_path = os.path.join(os.path.dirname(__file__), "custom_detectors", "custom_detectors.py")
        spec = importlib.util.spec_from_file_location("custom_detectors.custom_detectors", module_path)
        custom_detectors = importlib.util.module_from_spec(spec)

        # inject any user utility functions into the code automatically
        inject_imports = {
            "use_instruments": use_instruments,
            "non_blocking": non_blocking,
        }
        for name, mod in inject_imports.items():
            setattr(custom_detectors, name, mod)

        # load the module
        sys.modules["custom_detectors.custom_detectors"] = custom_detectors
        spec.loader.exec_module(custom_detectors)

        self.registry = {name: obj for name, obj
                         in inspect.getmembers(custom_detectors, inspect.isfunction)
                         if not name.startswith("_") and name not in forbidden_names}

        self.function_needs_headers = {}
        self.function_needs_kwargs = {}
        for name, obj in self.registry.items():
            self.function_needs_headers[name] = "headers" in inspect.signature(obj).parameters
            self.function_needs_kwargs[name] = "kwargs" in inspect.signature(obj).parameters


        # check if functions have requested user prometheus metrics
        for name, func in self.registry.items():
            target = get_underlying_function(func)
            if getattr(target, "prometheus_instruments", False):
                instruments = target.prometheus_instruments
                for instrument in instruments:
                    super().add_instrument(instrument)

        logger.info(f"Registered the following custom detectors: {self.registry.keys()}")


    def handle_request(self, content: str, detector_params: dict, headers: dict, **kwargs) -> List[ContentAnalysisResponse]:
        detections = []
        for custom_function_name in self.get_detection_functions_from_params(detector_params):
            if self.registry.get(custom_function_name):
                try:
                    func_headers = headers if self.function_needs_headers.get(custom_function_name) else None

                    if self.function_needs_kwargs.get(custom_function_name)and isinstance(detector_params[self.registry_name][custom_function_name], dict):
                        func_kwargs = detector_params[self.registry_name][custom_function_name]
                    else:
                        func_kwargs = None

                    with self.instrument_runtime(custom_function_name):
                        result = custom_func_wrapper(self.registry[custom_function_name], custom_function_name, content, func_headers, func_kwargs)
                    is_detection = result is not None
                    self.increment_detector_instruments(custom_function_name, is_detection)
                    if is_detection:
                        detections.append(result)
                except Exception as e:
                    self.throw_internal_detector_error(custom_function_name, logger, e, increment_requests=True)
            else:
                raise HTTPException(status_code=400, detail=f"Unrecognized custom function: {custom_function_name}")
        return detections
