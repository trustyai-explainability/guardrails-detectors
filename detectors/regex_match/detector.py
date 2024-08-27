import os
import re
import sys

sys.path.insert(0, os.path.abspath(".."))
from common.app import logger
from scheme import (
    ContentAnalysisHttpRequest,
    ContentAnalysisResponse,
    ContentsAnalysisResponse,
)


class Detector:
    def __init__(self, match_type="findall"):
        if match_type not in {"findall", "search", "match"}:
            raise ValueError(f"Invalid match_type: {match_type}")
        self.match_type = match_type
        logger.info(f"Initialized RegexDetector with match_type: {self.match_type}")

    def run(self, input: ContentAnalysisHttpRequest) -> ContentsAnalysisResponse:
        if not input.contents:
            logger.warning("No contents provided for analysis.")
            return ContentsAnalysisResponse(root=[])

        try:
            pattern = re.compile(input.regex_pattern)
        except re.error as e:
            logger.error(f"Invalid regex pattern: {e}")
            raise

        match_func = getattr(pattern, self.match_type, None)
        if not match_func:
            raise AttributeError(
                f"'re.Pattern' object has no attribute '{self.match_type}'"
            )

        contents_analyses = []

        for text in input.contents:
            content_analyses = []
            matches = match_func(text)
            content_analyses.append(
                ContentAnalysisResponse(
                    start=0,
                    end=len(text),
                    detection="has_regex_match",
                    detection_type="regex_match",
                    detection_value=bool(matches),
                    text=text,
                    matches=matches,
                    evidences=[],
                )
            )
            # Wrap `content_analysis` in a list to conform with the expected response structure
            contents_analyses.append(content_analyses)

        return contents_analyses
