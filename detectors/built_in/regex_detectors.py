import re
from time import time
from http.client import HTTPException
from typing import List
import logging
from base_detector_registry import BaseDetectorRegistry
from detectors.common.scheme import ContentAnalysisResponse

logger = logging.getLogger(__name__)

def email_address_detector(string: str) -> List[ContentAnalysisResponse]:
    """Detect email addresses in the text contents"""
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    return get_regex_detections(string, pattern, "pii", "email_address")

def credit_card_detector(string: str) -> List[ContentAnalysisResponse]:
    """Detect credit cards in the text contents (Visa, MasterCard, Amex, Discover, Diners Club, JCB) with Luhn check"""
    # Match major card types with separators (space or dash) between groups, not continuous digits
    pattern = (
        r"\b(?:"
        r"4\d{3}([- ])\d{4}\1\d{4}\1\d{4}"                # Visa 16-digit with separators
        r"|4\d{15}"                                       # Visa 16-digit continuous
        r"|5[1-5]\d{2}([- ])\d{4}\2\d{4}\2\d{4}"           # MasterCard 16-digit with separators
        r"|5[1-5]\d{14}"                                  # MasterCard 16-digit continuous
        r"|3[47]\d{2}([- ])\d{6}\3\d{5}"                   # Amex 15-digit with separators
        r"|3[47]\d{13}"                                   # Amex 15-digit continuous
        r"|6(?:011|5\d{2})([- ])\d{4}\4\d{4}\4\d{4}"       # Discover 16-digit with separators
        r"|6(?:011|5\d{2})\d{12}"                         # Discover 16-digit continuous
        r"|3(?:0[0-5]|[68]\d)\d([- ])\d{6}\5\d{4}"         # Diners Club 14-digit with separators
        r"|3(?:0[0-5]|[68]\d)\d{11}"                      # Diners Club 14-digit continuous
        r"|35\d{2}([- ])\d{4}\6\d{4}\6\d{4}"               # JCB 16-digit with separators
        r"|35\d{14}"                                      # JCB 16-digit continuous
        r")\b"
    )
    # Find all matches and filter with Luhn check
    detections = []
    for match in re.finditer(pattern, string):
        cc_number = match.group(0).replace(" ", "").replace("-", "")
        if is_luhn_valid(cc_number):
            detections.append(
                ContentAnalysisResponse(
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                    detection_type="pii",
                    detection="credit_card",
                    score=1.0
                )
            )
    return detections

def luhn_checksum(card_number: str):
    card_number = "".join(c for c in card_number if c in "0123456789")
    def digits_of(n):
        return [int(d) for d in str(n)]
    digits = digits_of(card_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = 0
    checksum += sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(d*2))
    return checksum % 10

def is_luhn_valid(card_number):
    return luhn_checksum(card_number) == 0


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
        super().__init__("regex")
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

    def handle_request(self, content: str, detector_params: dict, headers: dict) -> List[ContentAnalysisResponse]:
        detections = []
        for regex in self.get_detection_functions_from_params(detector_params):
            new_detections = []
            try:
                if regex == "$CUSTOM_REGEX":
                    continue
                elif regex in self.registry:
                    func_name = regex
                    with self.instrument_runtime(func_name):
                        new_detections = self.registry[regex](content)
                else:
                    func_name = "custom_regex" # don't publish custom regexes to prometheus labels, to limit metric cardinality
                    with self.instrument_runtime(func_name):
                        new_detections += get_regex_detections(content, regex, "regex", "custom-regex")
            except Exception as e:
                print(e)
                self.throw_internal_detector_error(func_name, logger, e, increment_requests=True)
            self.increment_detector_instruments(func_name, len(new_detections) > 0)
            detections += new_detections
        return detections
