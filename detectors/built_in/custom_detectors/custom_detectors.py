"""
This is an example custom_detectors.py file. Overwrite this file to define custom guardrailing
logic!

See [docs/custom_detectors.md](../../docs/custom_detectors.md) for more details.
"""

import time
def slow_func(text: str) -> bool:
    time.sleep(.25)
    return False
    
from prometheus_client import Counter

prompt_rejection_counter = Counter(
    "trustyai_guardrails_system_prompt_rejections",
    "Number of rejections by the system prompt",
)
  
@use_instruments(instruments=[prompt_rejection_counter])  
def has_metrics(text: str) -> bool:
    if "sorry" in text:
        prompt_rejection_counter.inc()
    return False
    
background_metric = Counter(
    "trustyai_guardrails_background_metric",
    "Runs some logic in the background without blocking the /detections call"
)
@use_instruments(instruments=[background_metric])
@non_blocking(return_value=False)
def background_function(text: str) -> bool:
    time.sleep(.25)
    if "sorry" in text:
        background_metric.inc()     
    return False
