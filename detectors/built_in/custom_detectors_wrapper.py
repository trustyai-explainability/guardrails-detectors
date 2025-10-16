import ast
import os
import traceback

from fastapi import HTTPException
import inspect
import logging
from typing import List, Optional, Callable


from base_detector_registry import BaseDetectorRegistry
from detectors.common.scheme import ContentAnalysisResponse

logger = logging.getLogger(__name__)

def custom_func_wrapper(func: Callable, func_name: str, s: str, headers: dict) -> Optional[ContentAnalysisResponse]:
    """Convert a some f(text)->bool into a Detector response"""
    sig = inspect.signature(func)
    try:
        if headers is not None:
            result = func(s, headers)
        else:
            result = func(s)

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

        issues = static_code_analysis(module_path = os.path.join(os.path.dirname(__file__), "custom_detectors", "custom_detectors.py"))
        if issues:
            logging.error(f"Detected {len(issues)} potential security issues inside the custom_detectors file: {issues}")
            raise ImportError(f"Unsafe code detected in custom_detectors:\n" + "\n".join(issues))

        import custom_detectors.custom_detectors as custom_detectors

        self.registry = {name: obj for name, obj
                         in inspect.getmembers(custom_detectors, inspect.isfunction)
                         if not name.startswith("_")}
        self.function_needs_headers = {name: "headers" in inspect.signature(obj).parameters for name, obj in self.registry.items() }
        logger.info(f"Registered the following custom detectors: {self.registry.keys()}")


    def handle_request(self, content: str, detector_params: dict, headers: dict, **kwargs) -> List[ContentAnalysisResponse]:
        detections = []
        for custom_function_name in self.get_detection_functions_from_params(detector_params):
            if self.registry.get(custom_function_name):
                try:
                    func_headers = headers if self.function_needs_headers.get(custom_function_name) else None
                    with self.instrument_runtime(custom_function_name):
                        result = custom_func_wrapper(self.registry[custom_function_name], custom_function_name, content, func_headers)
                    is_detection = result is not None
                    self.increment_detector_instruments(custom_function_name, is_detection)
                    if is_detection:
                        detections.append(result)
                except Exception as e:
                    self.throw_internal_detector_error(custom_function_name, logger, e, increment_requests=True)
            else:
                raise HTTPException(status_code=400, detail=f"Unrecognized custom function: {custom_function_name}")
        return detections
