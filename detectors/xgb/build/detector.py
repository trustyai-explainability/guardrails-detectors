import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import pathlib
import torch
import xgboost as xgb
from detectors.common.scheme import (
    ContentAnalysisHttpRequest,
    ContentAnalysisResponse,
)
import pickle as pkl
from base_detector_registry import BaseDetectorRegistry

try:
    from common.app import logger
except ImportError:
    sys.path.insert(0, os.path.join(pathlib.Path(__file__).parent.parent.resolve()))
    from common.app import logger

class Detector:
    def __init__(self):
        # initialize the detector
        model_files_path = os.path.abspath(os.path.join(os.sep, "app", "model_artifacts"))
        if not os.path.exists(model_files_path):
            model_files_path = os.path.join("build", "model_artifacts")
        logger.info(model_files_path)

        self.model = pkl.load(open(os.path.join(model_files_path, 'model.pkl'), 'rb'))
        self.vectorizer = pkl.load(open(os.path.join(model_files_path, 'vectorizer.pkl'), 'rb'))

        if torch.cuda.is_available():
            self.cuda_device = torch.device("cuda")
            torch.cuda.empty_cache()
            self.model.to(self.cuda_device)
            logger.info("cuda_device".upper() + " " + str(self.cuda_device))
            self.batch_size = 1
        else:
            self.batch_size = 8
        logger.info("Detector initialized.")

    def run(self, request: ContentAnalysisHttpRequest) -> ContentAnalysisResponse:
        if hasattr(request, "detection_type") and request.detection_type != "spamCheck":
            logger.warning(f"Unsupported detection type: {request.detection_type}")

        content_analyses = []
        for batch_idx in range(0, len(request.contents), self.batch_size):
            text = request.contents[batch_idx:batch_idx + self.batch_size]
            vectorized_text = self.vectorizer.transform(text)
            predictions = self.model.predict(vectorized_text)
            detections = any([True for p in predictions if p == 1])

        content_analyses.append(
            ContentAnalysisResponse(
                start=0,
                end=len(text),
                detection=detections,
                detection_type="spamCheck",
                text=text,
                evidences=[],
            )
        )
        return content_analyses
