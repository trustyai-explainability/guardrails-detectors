# Custom Detectors
You can overwrite [custom_detectors.py](detectors/built_in/custom_detectors/custom_detectors.py) to create
custom detectors in the built-in server based off of arbitrary Python code. This lets you quickly and flexible
create your own detection logic!

The following rules apply:
1) Each function defined in the `custom_detectors.py` file (except for those starting with "_") will be registered as a detector
2) Functions that accept a parameter `headers` will receive the inbound request headers as a parameter
   * see the `function_that_needs_headers` example in [custom_detectors.py](detectors/built_in/custom_detectors/custom_detectors.py) for usage
3) Functions that are intended to be used as detectors must either return a `bool` or a `dict`: 
   1) Return values that evaluate to false (e.g., `{}`, `""`, `None`, etc) are treated as non-detections 
   2) Boolean responses of `true` are considered a detection 
      * see the `over_100_characters` example in [custom_detectors.py](detectors/built_in/custom_detectors/custom_detectors.py) for usage
   3) Dict response that are parseable as a `ContentAnalysisResponse` object are considered a detection
      * see the `contains_word` example in [custom_detectors.py](detectors/built_in/custom_detectors/custom_detectors.py) for usage
4) This code may not import `os`, `subprocess`, `sys`, or `shutil` for security reasons
5) This code may not call `eval`, `exec`, `open`, `compile`, or `input` for security reasons


## Utility Decorators
The following decorators are also available, and are automatically imported into the custom_detectors.py file:

### `@use_instruments(instruments=[$INSTRUMENT_1, ..., $INSTRUMENT_N])`
 Use this decorator to register your own Prometheus instruments with the server's main
 `/metrics` registry. See the `function_that_has_prometheus_metrics` example 
 in [custom_detectors.py](detectors/built_in/custom_detectors/custom_detectors.py) for usage.

### `@non_blocking(return_value=$RETURN_VALUE)`
Use this decorator to indicate that the logic inside this function should run in a non-blocking
background thread. The guardrail function will immediately return $RETURN_VALUE while launching 
your function logic into a background thread.

This enables a number of use-cases, such as:
* Producing some background analysis metric over the input/output, without adding latency to the system
* Performing "silent" guardrailing, e.g., adding information to a server or alerting admins

See the `background_function` example in
[custom_detectors.py](detectors/built_in/custom_detectors/custom_detectors.py) for usage.

## More Examples
For a "real-world" example, check out the [TrustyAI custom detectors demo](https://github.com/trustyai-explainability/trustyai-llm-demo/blob/main/custom-detectors/custom_detectors.py)!