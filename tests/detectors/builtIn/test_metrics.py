import enum
import os
import time

import pytest
from starlette.testclient import TestClient


# DO NOT IMPORT THIS VALUE, if we import common.app before the test fixtures we can break prometheus multiprocessing
METRIC_PREFIX = "trustyai_guardrails"

CUSTOM_DETECTORS_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../detectors/built_in/custom_detectors/custom_detectors.py"
)

with open(CUSTOM_DETECTORS_PATH) as f:
    ORIGINAL_CODE = f.read()

METRIC_TEST_CODE = f'''\
def throws_error(text: str) -> bool:
    if text == "illegal":
        return True
    elif text == "error":
        raise ValueError(text)
    else:
        return False
        
import time
def slow_func(text: str) -> bool:
    time.sleep(.25)
    return False
    
from prometheus_client import Counter

prompt_rejection_counter = Counter(
    "{METRIC_PREFIX}_system_prompt_rejections",
    "Number of rejections by the system prompt",
)
  
@use_instruments(instruments=[prompt_rejection_counter])  
def has_metrics(text: str) -> bool:
    if "sorry" in text:
        prompt_rejection_counter.inc()
    return False
    
background_metric = Counter(
    "{METRIC_PREFIX}_background_metric",
    "Runs some logic in the background without blocking the /detections call"
)
@use_instruments(instruments=[background_metric])
@non_blocking(return_value=False)
def background_function(text: str) -> bool:
    time.sleep(.25)
    if "sorry" in text:
        background_metric.inc()     
    return False
'''


def write_code_to_custom_detectors(code: str):
    with open(CUSTOM_DETECTORS_PATH, "w") as f:
        f.write(code)

def restore_original_code():
    write_code_to_custom_detectors(ORIGINAL_CODE)


def wait_for_metric(client, metric_name, expected_value, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        metric_dict = get_metric_dict(client)
        if metric_dict.get(metric_name, 0) == expected_value:
            return True
        time.sleep(0.1)
    raise AssertionError(f"Metric {metric_name} did not reach {expected_value} in {timeout} seconds")

class DetectorOutcome(enum.Enum):
    PASS = 0
    FLAG = 1
    ERROR = 2


def send_filetype_request(desired_outcome: DetectorOutcome, client: TestClient):
    if desired_outcome == DetectorOutcome.PASS:
        payload = {
            "contents": ['{"a": 1, "b": 2}'],
            "detector_params": {"file_type": ["json"]}
        }
    elif desired_outcome == DetectorOutcome.FLAG:
        payload = {
            "contents": ['{"a": 1, "b": 2'],
            "detector_params": {"file_type": ["json"]}
        }
    else:
        # file type validator failures ARE flags,
        raise ValueError("Filetype requests cannot be induced to raise errors")
    client.post("/api/v1/text/contents", json=payload)


def send_regex_request(desired_outcome: DetectorOutcome, client: TestClient):
    if desired_outcome == DetectorOutcome.PASS:
        payload = {
            "contents": ["totally innocuous"],
            "detector_params": {"regex": ["\b(?i:orange|apple|cranberry|pineapple|grape)\b"]}
        }
        expected_status_code = 200
    elif desired_outcome == DetectorOutcome.FLAG:
        payload = {
            "contents": ["orange and apple and cranberry"],
            "detector_params": {"regex": [r"\b(?i:orange|apple|cranberry|pineapple|grape)\b"]}
        }
        expected_status_code =200
    else:
        payload = {
            "contents": ["totally innocuous"],
            "detector_params": {"regex": ["["]}
        }
        expected_status_code = 500
    response = client.post("/api/v1/text/contents", json=payload)
    assert response.status_code == expected_status_code



def send_custom_request(desired_outcome: DetectorOutcome, client: TestClient):
    if desired_outcome == DetectorOutcome.PASS:
         payload = {
            "contents": ["fine"],
            "detector_params": {"custom": ["throws_error"]}
        }
         expected_status_code = 200
    elif desired_outcome == DetectorOutcome.FLAG:
        payload = {
            "contents": ["illegal"],
            "detector_params": {"custom": ["throws_error"]}
        }
        expected_status_code = 200
    else:
        payload = {
            "contents": ["error"],
            "detector_params": {"custom": ["throws_error"]}
        }
        expected_status_code = 500
    response = client.post("/api/v1/text/contents", json=payload)
    assert response.status_code == expected_status_code


def get_metric_dict(client: TestClient):
    metrics = client.get("/metrics")
    metrics = metrics.content.decode().split("\n")
    metric_dict = {}

    for m in metrics:
        if METRIC_PREFIX in m and "HELP" not in m and "TYPE" not in m:
            key, value = m.split(" ")
            metric_dict[key] = float(value)

    return metric_dict


class TestMetrics:
    @pytest.fixture
    def client(self):
        write_code_to_custom_detectors(METRIC_TEST_CODE)

        from detectors.built_in.app import app

        # clear the metric registry at the start of each test, but AFTER the multiprocessing metrics is set up
        import prometheus_client
        prometheus_client.REGISTRY._names_to_collectors.clear()

        from detectors.built_in.custom_detectors_wrapper import CustomDetectorRegistry
        from detectors.built_in.regex_detectors import RegexDetectorRegistry
        from detectors.built_in.file_type_detectors import FileTypeDetectorRegistry

        for detector_registry in [
            RegexDetectorRegistry(),
            FileTypeDetectorRegistry(),
            CustomDetectorRegistry()
        ]:
            app.set_detector(detector_registry, detector_registry.registry_name)
            detector_registry.set_instruments(app.state.instruments)
        return TestClient(app)

    @pytest.fixture(autouse=True)
    def cleanup_custom_detectors(self):
        yield
        restore_original_code()



    def test_prometheus(self, client: TestClient):
        # # send 40% passing filetype requests
        for _ in range(4):
            send_filetype_request(desired_outcome=DetectorOutcome.PASS, client=client)
        for _ in range(6):
            send_filetype_request(desired_outcome=DetectorOutcome.FLAG, client=client)

        # send 20% passing, 70% failing, 10% erroring file_type requests
        for _ in range(2):
            send_regex_request(desired_outcome=DetectorOutcome.PASS, client=client)
        for _ in range(7):
            send_regex_request(desired_outcome=DetectorOutcome.FLAG, client=client)
        send_regex_request(desired_outcome=DetectorOutcome.ERROR, client=client)

        # send 50% passing, 30% failing, 20$ erroring custom function requests
        for _  in range(50):
            send_custom_request(desired_outcome=DetectorOutcome.PASS, client=client)
        for _ in range(30):
            send_custom_request(desired_outcome=DetectorOutcome.FLAG, client=client)
        for _ in range(20):
            send_custom_request(desired_outcome=DetectorOutcome.ERROR, client=client)


        expected_results = {
            f'{METRIC_PREFIX}_detections_total{{detector_kind="regex",detector_name="custom_regex"}}': 7.0,
            f'{METRIC_PREFIX}_errors_total{{detector_kind="regex",detector_name="custom_regex"}}': 1.0,
            f'{METRIC_PREFIX}_requests_total{{detector_kind="regex",detector_name="custom_regex"}}': 10.0,

            f'{METRIC_PREFIX}_detections_total{{detector_kind="file_type",detector_name="json"}}': 6.0,
            f'{METRIC_PREFIX}_errors_total{{detector_kind="file_type",detector_name="json"}}': 0.0,
            f'{METRIC_PREFIX}_requests_total{{detector_kind="file_type",detector_name="json"}}': 10.0,

            f'{METRIC_PREFIX}_detections_total{{detector_kind="custom",detector_name="throws_error"}}': 30.0,
            f'{METRIC_PREFIX}_errors_total{{detector_kind="custom",detector_name="throws_error"}}': 20.0,
            f'{METRIC_PREFIX}_requests_total{{detector_kind="custom",detector_name="throws_error"}}': 100.0,
        }

        metric_dict = get_metric_dict(client)
        print(metric_dict)

        for expected_key, expected_val in expected_results.items():
            assert expected_key in metric_dict, f"expected key {expected_key} not found in metric dict"
            assert metric_dict[expected_key] == expected_val,  f"metric {expected_key} value={metric_dict[expected_key]} did not match expected value {expected_val}"


    def test_runtime_metrics(self, client: TestClient):
        payload = {
            "contents": ["totally innocuous"],
            "detector_params": {"custom": ["slow_func"]}
        }
        # 8 calls of this function should induce ~ 2 seconds of latency
        for _ in range(8):
            client.post("/api/v1/text/contents", json=payload)
        metric_dict = get_metric_dict(client)

        func_runtime = metric_dict[f'{METRIC_PREFIX}_runtime_total{{detector_kind="custom",detector_name="slow_func"}}']
        assert func_runtime > 1.8
        assert func_runtime < 2.2


    def test_user_metrics(self, client: TestClient):
        # ensure that metrics created by the user are visible
        for i in range(8):
            payload = {
                "contents": ["sorry I can't help"] if i%2==0 else ["a different response"],
                "detector_params": {"custom": ["has_metrics"]}
            }
            client.post("/api/v1/text/contents", json=payload)
        metric_dict = get_metric_dict(client)
        print(metric_dict)
        assert metric_dict[f'{METRIC_PREFIX}_detections_total{{detector_kind="custom",detector_name="has_metrics"}}'] == 0
        assert metric_dict[f"{METRIC_PREFIX}_system_prompt_rejections_total"] == 4


    def test_non_blocking_metrics(self, client: TestClient):
        # ensure that metrics created by the user are visible
        start_time = time.time()
        for i in range(100):
            payload = {
                "contents": ["sorry I can't help"] if i%2==0 else ["a different response"],
                "detector_params": {"custom": ["background_function"]}
            }
            client.post("/api/v1/text/contents", json=payload)
        call_time = time.time()

        # each call to background function contains a `time.sleep(.25)` call - if the function logic is
        # correctly sent to a background thread, these calls should take much less than 100 * .25 = 25 seconds
        assert call_time - start_time < 0.5

        # this will wait for the long-running background threads to finish
        wait_for_metric(client, f"{METRIC_PREFIX}_background_metric_total", 50)
        metric_dict = get_metric_dict(client)

        # check that the reported runtime of the functions matches the _call_ time, not the calculation time
        func_runtime = metric_dict[f'{METRIC_PREFIX}_runtime_total{{detector_kind="custom",detector_name="background_function"}}']
        assert func_runtime < 0.5
