from abc import ABC, abstractmethod
from typing import List

from guardrails_detectors_common import ContentAnalysisResponse

class BaseDetectorRegistry(ABC):
    def __init__(self):
        self.registry = None

    @abstractmethod
    def handle_request(self, content: str, detector_params: dict) -> List[ContentAnalysisResponse]:
        pass
    
    def get_registry(self):
        return self.registry