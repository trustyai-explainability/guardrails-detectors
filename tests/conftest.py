import os
import sys
import pytest


@pytest.fixture(autouse=True)
def setup_imports():
    """Setup Python path for imports"""
    project_root = os.path.dirname(os.path.dirname(__file__))
    src_path = os.path.join(project_root, "src")
    detectors_path = os.path.join(src_path, "guardrails_detectors")
    huggingface_path = os.path.join(detectors_path, "huggingface")
    llm_judge_path = os.path.join(detectors_path, "llm_judge")
    built_in_detectors_path = os.path.join(detectors_path, "built_in")
    common_path = os.path.join(detectors_path, "common")
    
    paths = [
        src_path,
        detectors_path,
        huggingface_path,
        llm_judge_path,
        built_in_detectors_path,
        common_path,
        project_root,
    ]

    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"Added to sys.path: {path}")

