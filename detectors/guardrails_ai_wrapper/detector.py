import os
import sys

sys.path.insert(0, os.path.abspath(".."))
from common.app import logger
from scheme import (
    ContentAnalysisHttpRequest,
    ContentAnalysisResponse,
    ContentsAnalysisResponse,
)

from guardrails import Guard, OnFailAction
from guardrails.hub import RegexMatch, CompetitorCheck, ToxicLanguage


class Detector:
    def __init__(self):
        self.guard = Guard().use_many(
            RegexMatch(regex=r"^(?!.*\bpotatoe\b).*$", on_fail=OnFailAction.EXCEPTION),
            CompetitorCheck(
                ["Apple", "Microsoft", "Google"], on_fail=OnFailAction.EXCEPTION
            ),
            ToxicLanguage(
                threshold=0.5,
                validation_method="sentence",
                on_fail=OnFailAction.EXCEPTION,
            ),
        )
        logger.info("Guardrails AI Wrapper initialized")

    def run(self, input: ContentAnalysisHttpRequest) -> ContentsAnalysisResponse:
        contents_analyses = []
        for text in input.contents:
            content_analyses = []
            try:
                logger.info(f"Validating text: {text}")
                validation_result = self.guard.validate(text)
                status = "success"
                logger.info(f"Validation successful for text: {text}")
            except Exception as e:
                logger.error(f"Validation failed for text: {text} with error: {e}")
                validation_result = str(e)
                status = "failed"

            content_analyses.append(
                ContentAnalysisResponse(
                    detection_type="guardrails_ai",
                    status=status,
                    start=0,
                    end=len(text),
                    text=text,
                    validation_result=validation_result,
                    evidences=[],
                )
            )
            contents_analyses.append(content_analyses)
        logger.info("Content analysis completed")
        return contents_analyses
