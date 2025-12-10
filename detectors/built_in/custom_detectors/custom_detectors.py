"""
This is an example custom_detectors.py file. Overwrite this file to define custom guardrailing
logic!

See [docs/custom_detectors.md](../../docs/custom_detectors.md) for more details.
"""

# example boolean-returning function
def over_100_characters(text: str) -> bool:
    return len(text)>100

# example dict-returning function
def contains_word(text: str) -> dict:
    detection = "apple" in text.lower()
    if detection:
        detection_position = text.lower().find("apple")
        return {
            "start":detection_position,  # start position of detection in text
            "end": detection_position+5, # end position of detection in text
            "text": text, # "the flagged text, or some arbitrary message to return to the user"
            "detection_type": "content_check", #detection_type -> use these fields to define your detector taxonomy as you see fit
            "detection": "forbidden_word: apple", ##detection -> use these fields to define your detector taxonomy as you see fit
            "score": 1.0 # the score/severity/probability of the detection
        }
    else:
        return {}

def _this_function_will_not_be_exposed():
    pass

def function_that_needs_headers(text: str, headers: dict) -> bool:
    return headers['magic-key'] != "123"

def function_that_needs_kwargs(text: str, **kwargs: dict) -> bool:
    return kwargs['magic-key'] != "123"

# === CUSTOM METRICS =====
import time
from prometheus_client import Counter
prompt_rejection_counter = Counter(
    "system_prompt_rejections",
    "Number of rejections by the system prompt",
)
@use_instruments(instruments=[prompt_rejection_counter])
def has_metrics(text: str) -> bool:
    if "sorry" in text:
        prompt_rejection_counter.inc()
    return False


background_metric = Counter(
    "background_metric",
    "Runs some logic in the background without blocking the /detections call"
)
@use_instruments(instruments=[background_metric])
@non_blocking(return_value=False)
def background_function(text: str) -> bool:
    time.sleep(.25)
    if "sorry" in text:
        background_metric.inc()
    return False