"""
This is an example custom_detectors.py file. Here, you can define any arbitrary Python code as a
Guardrail detector.

The following rules apply:
1) Each function defined in this file (except for those starting with "_") will be registered as a detector
2) Functions that accept a parameter "headers" will receive the inbound request headers as a parameter
3) Functions may either return a boolean or a dict:
    3a) Return values that evaluate to false (e.g., {}, "", None, etc) are treated as non-detections
    3b) Boolean responses of "true" are considered a detection
    3c) Dict response must be parseable as a ContentAnalysisResponse object (see example below)
4) This code may not import "os", "subprocess", "sys", or "shutil" for security reasons
5) This code may not call "eval", "exec", "open", "compile", or "input" for security reasons
"""

# example boolean-returning function
def over_100_characters(text: str) -> bool:
    return len(text)>100

# example dict-returning function
def contains_word(text: str) -> dict:
    detection = "apple" in text.lower()
    if detection:
        detection_position = text.find("apple")
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

