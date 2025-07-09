import re
from http.client import HTTPException
from typing import List
from base_detector_registry import BaseDetectorRegistry
from detectors.common.scheme import ContentAnalysisResponse


def email_address_detector(string: str) -> List[ContentAnalysisResponse]:
    """Detect email addresses in the text contents"""
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    return get_regex_detections(string, pattern, "pii", "email_address")

def credit_card_detector(string: str) -> List[ContentAnalysisResponse]:
    """Detect credit cards in the text contents"""
    pattern = r"\b(?:4\d{3}|5[0-5]\d{2}|6\d{3}|1\d{3}|3\d{3})[- ]\d{4}[- ]\d{4}[- ]\d{4}\b"
    return get_regex_detections(string, pattern, "pii", "credit_card")


def ipv4_detector(string: str) -> List[ContentAnalysisResponse]:
    """Detect IPv4 addresses in the text contents"""
    pattern = re.compile(
        u"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
        re.IGNORECASE,
    )
    return get_regex_detections(string, pattern, "pii", "ipv4")

def ipv6_detector(string: str) -> List[ContentAnalysisResponse]:
    """Detect IPv6 addresses in the text contents"""
    pattern = re.compile(
        u"\s*(?!.*::.*::)(?:(?!:)|:(?=:))(?:[0-9a-f]{0,4}(?:(?<=::)|(?<!::):)){6}(?:[0-9a-f]{0,4}(?:(?<=::)|(?<!::):)[0-9a-f]{0,4}(?:(?<=::)|(?<!:)|(?<=:)(?<!::):)|(?:25[0-4]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-4]|2[0-4]\d|1\d\d|[1-9]?\d)){3})\s*",
        re.VERBOSE | re.IGNORECASE | re.DOTALL,
    )
    return get_regex_detections(string, pattern, "pii", "ipv6")

# === USA Specific =================================================================================
def ssn_detector(string: str) -> List[ContentAnalysisResponse]:
    """Detect social security numbers in the text contents"""
    pattern = r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"
    return get_regex_detections(string, pattern, "pii", "social_security_number")

def us_phone_number_detector(string: str) -> List[ContentAnalysisResponse]:
    """Detect US phone numbers in the text contents"""
    pattern = r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]+\d{3}[-.\s]?\d{4}\b"
    return get_regex_detections(string, pattern, "pii", "us-phone-number")

# === UK Specific =================================================================================
def uk_post_code_detector(string: str) -> List[ContentAnalysisResponse]:
    """Detect UK post codes in the text contents"""
    pattern = r"\b([A-Z]{1,2}[0-9][0-9A-Z]? ?[0-9][A-Z]{2})\b"
    return get_regex_detections(string, pattern, "pii", "uk-post-code")


def get_regex_detections(string, pattern, detection_type, detection) -> List[ContentAnalysisResponse]:
    detections = []
    for match in re.finditer(pattern, string):
        detections.append(
            ContentAnalysisResponse(
                start=match.start(),
                end=match.end(),
                text=match.string[match.start():match.end()],
                detection_type=detection_type,
                detection=detection,
                score=1.0
            ))
    return detections

# dummy function to add documention on the custom regex detector to the registr
def custom_regex_documenter():
    """Replace $CUSTOM_REGEX with a custom regex to define your own regex detector"""


# === ROUTER =======================================================================================
class RegexDetectorRegistry(BaseDetectorRegistry):
    def __init__(self):
        self.registry =  {
            "credit-card": credit_card_detector,
            "email": email_address_detector,
            "ipv4": ipv4_detector,
            "ipv6": ipv6_detector,
            "us-phone-number": us_phone_number_detector,
            "us-social-security-number": ssn_detector,
            "uk-post-code": uk_post_code_detector,
            "$CUSTOM_REGEX": custom_regex_documenter,
        }

    def handle_request(self, content: str, detector_params: dict) -> List[ContentAnalysisResponse]:
        detections = []
        if "regex" in detector_params and isinstance(detector_params["regex"], list):
            for regex in detector_params["regex"]:
                if regex == "$CUSTOM_REGEX":
                    pass
                elif regex in self.registry:
                    detections += self.registry[regex](content)
                else:
                    detections += get_regex_detections(content, regex, "regex", "custom-regex")
        return detections
