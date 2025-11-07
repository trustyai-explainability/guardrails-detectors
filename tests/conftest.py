import os
import shutil
import sys
import pytest
import tempfile


@pytest.fixture(autouse=True)
def setup_imports():
    """Setup Python path for imports"""
    project_root = os.path.dirname(os.path.dirname(__file__))
    detectors_path = os.path.join(project_root, "detectors")
    huggingface_path = os.path.join(detectors_path, "huggingface")
    llm_judge_path = os.path.join(detectors_path, "llm_judge")
    built_in_detectors_path = os.path.join(detectors_path, "built_in")
    paths = [
        huggingface_path,
        detectors_path,
        project_root,
        llm_judge_path,
        built_in_detectors_path,
    ]

    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"Added to sys.path: {path}")

@pytest.fixture(scope="session", autouse=True)
def prometheus_multiproc_dir():
    """
    Create a temporary directory for PROMETHEUS_MULTIPROC_DIR and set the environment variable.
    """
    tmpdir = tempfile.mkdtemp(prefix="prometheus_multiproc_")
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = tmpdir
    yield tmpdir
    # Cleanup will be handled by the next fixture

@pytest.fixture(scope="session", autouse=True)
def cleanup_prometheus_multiproc_dir(request, prometheus_multiproc_dir):
    """
    Cleanup the PROMETHEUS_MULTIPROC_DIR after the test session.
    """
    yield
    shutil.rmtree(prometheus_multiproc_dir, ignore_errors=True)
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        del os.environ["PROMETHEUS_MULTIPROC_DIR"]