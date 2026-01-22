import importlib
import sys
from http.client import HTTPException

import pytest
import os
from fastapi.testclient import TestClient


CUSTOM_DETECTORS_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../detectors/built_in/custom_detectors/custom_detectors.py"
)

with open(CUSTOM_DETECTORS_PATH) as f:
    SAFE_CODE = f.read()

UNSAFE_CODE = '''
import os
def evil(text: str) -> bool:
    os.system("echo haha gottem")
    return True
'''

UNSAFE_CODE_IMPORT_FROM = '''
from sys import path
def func(text: str) -> bool:
    return True
'''

SAFE_CODE_IMPORT_FROM_ENVIRON = '''
from os import environ
def func(text: str) -> bool:
    return True
'''

def write_code_to_custom_detectors(code: str):
    with open(CUSTOM_DETECTORS_PATH, "w") as f:
        f.write(code)

def restore_safe_code():
    write_code_to_custom_detectors(SAFE_CODE)


class TestCustomDetectors:
    @pytest.fixture
    def client(self):
        from detectors.built_in.app import app

        # clear the metric registry at the start of each test, but AFTER the multiprocessing metrics is set up
        import prometheus_client
        prometheus_client.REGISTRY._names_to_collectors.clear()

        from detectors.built_in.custom_detectors_wrapper import CustomDetectorRegistry
        app.set_detector(CustomDetectorRegistry(), "custom")
        return TestClient(app)

    @pytest.fixture(autouse=True)
    def cleanup_custom_detectors(self):
        # Always restore safe code after test
        yield
        restore_safe_code()

    def test_missing_detector_type(self, client):
        payload = {
            "contents": ["What is an apple?"],
            "detector_params": {"custom1": ["contains_word"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 400 and "Detector custom1 not found" in resp.text


    def test_custom_detectors(self, client):
        payload = {
            "contents": ["What is an apple?"],
            "detector_params": {"custom": ["contains_word"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        texts = [d["text"] for d in resp.json()[0]]
        assert "What is an apple?" in texts

    def test_custom_detectors_not_match(self, client):
        msg = "What is an banana?"
        payload = {
            "contents": [msg],
            "detector_params": {"custom": ["contains_word"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        texts = [d["text"] for d in resp.json()[0]]
        assert msg not in texts

    def test_custom_detectors_need_header(self, client):
        msg = "What is an banana?"
        payload = {
            "contents": [msg],
            "detector_params": {"custom": ["function_that_needs_headers"]}
        }

        # shouldn't flag
        headers = {"magic-key": "123"}
        resp = client.post("/api/v1/text/contents", json=payload, headers=headers)
        assert resp.status_code == 200
        texts = [d["text"] for d in resp.json()[0]]
        assert msg not in texts

        # should flag
        headers = {"magic-key": "wrong"}
        resp = client.post("/api/v1/text/contents", json=payload, headers=headers)
        assert resp.status_code == 200
        texts = [d["text"] for d in resp.json()[0]]
        assert msg in texts

    def test_custom_detectors_need_kwargs(self, client):
        msg = "What is an banana?"
        payload1 = {
            "contents": [msg],
            "detector_params": {"custom": {"function_that_needs_kwargs": {"magic-key": "123"}}}
        }
        payload2 = {
            "contents": [msg],
            "detector_params": {"custom": {"function_that_needs_kwargs": {"magic-key": "345"}}}
        }

        # shouldn't flag
        resp = client.post("/api/v1/text/contents", json=payload1)
        assert resp.status_code == 200
        texts = [d["text"] for d in resp.json()[0]]
        assert msg not in texts

        # should flag
        resp = client.post("/api/v1/text/contents", json=payload2)
        assert resp.status_code == 200
        texts = [d["text"] for d in resp.json()[0]]
        assert msg in texts


    def test_unsafe_code(self, client):
        write_code_to_custom_detectors(UNSAFE_CODE)
        from detectors.built_in.custom_detectors_wrapper import CustomDetectorRegistry
        with pytest.raises(ImportError) as excinfo:
            CustomDetectorRegistry()
        assert "Unsafe code detected" in str(excinfo.value)
        assert "Forbidden import: os" in str(excinfo.value) or "os.system" in str(excinfo.value)


    def test_unsafe_code_import_from(self, client):
        write_code_to_custom_detectors(UNSAFE_CODE_IMPORT_FROM)
        from detectors.built_in.custom_detectors_wrapper import CustomDetectorRegistry
        with pytest.raises(ImportError) as excinfo:
            CustomDetectorRegistry()
        assert "Unsafe code detected" in str(excinfo.value)
        assert "Forbidden import: sys" in str(excinfo.value) or "sys.path" in str(excinfo.value)


    def test_safe_code_import_from_environ(self, client):
        # from os import environ <- should not trigger the unsafe import error
        write_code_to_custom_detectors(SAFE_CODE_IMPORT_FROM_ENVIRON)
        from detectors.built_in.custom_detectors_wrapper import CustomDetectorRegistry
        CustomDetectorRegistry()
        assert True


    def test_custom_detectors_func_doesnt_exist(self, client):
        payload = {
            "contents": ["What is an apple?"],
            "detector_params": {"custom": ["abc"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 400 and "Unrecognized custom function: abc" in resp.text
