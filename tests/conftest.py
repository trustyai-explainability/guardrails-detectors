import os
import sys
import pytest


@pytest.fixture(autouse=True)
def setup_imports():
    """Setup Python path for imports"""
    project_root = os.path.dirname(os.path.dirname(__file__))
    detectors_path = os.path.join(project_root, "detectors")
    huggingface_path = os.path.join(detectors_path, "huggingface")
    paths = [
        huggingface_path,
        detectors_path,
        project_root,
    ]

    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"Added to sys.path: {path}")
