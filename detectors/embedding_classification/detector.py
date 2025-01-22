import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(""))
# from common.scheme import TextDetectionHttpRequest, TextDetectionResponse
import os
import pathlib
import pickle as pkl
import numpy as np
import torch
import torch.nn


from scheme import (
    ContentAnalysisHttpRequest,
    ContentAnalysisResponse,
    ContentsAnalysisResponse,
    EvidenceObj
)

# Detector imports
from sentence_transformers import SentenceTransformer

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

        self.model = SentenceTransformer(os.path.join(model_files_path, "dunzhang", "stella_en_1"), trust_remote_code=True)
        self.reducer = pkl.load(open(os.path.join(model_files_path, "umap.pkl"), "rb"))
        self.centroids = pd.read_pickle(os.path.join(model_files_path, "centroids.pkl"))

        if torch.cuda.is_available():
            # transparently taking a cuda gpu for an actor
            self.cuda_device = torch.device("cuda")
            torch.cuda.empty_cache()
            self.model.to(self.cuda_device)
            # self.tokenizer.to(self.cuda_device)
            # AttributeError: 'RobertaTokenizerFast' object has no attribute 'to'
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            logger.info("cuda_device".upper() + " " + str(self.cuda_device))
            self.batch_size = 1
        else:
            self.batch_size = 8

        logger.info("Detector initialized.")

    def get_distance_to_centroids(self, point):
        dists = self.centroids.apply(lambda x: np.linalg.norm(point-x), 1).sort_values().iloc[:10]
        return {k:np.round(v/1, 2) for k,v in dists.to_dict().items()}


    def run(self, request: ContentAnalysisHttpRequest) -> ContentsAnalysisResponse:
        # run the classification for each entry on contents array
        # logger.info(tokenizer_parameters)
        contents_analyses = []
        for batch_idx in range(0, len(request.contents), self.batch_size):
            texts = request.contents[batch_idx:batch_idx+self.batch_size]
            embedding = self.model.encode(texts, prompt_name="s2p_query")
            umapped = self.reducer.transform(embedding)

            for idx in range(len(umapped)):
                topics = self.get_distance_to_centroids(umapped[idx])

                logger.debug(topics)
                matched_topics = {k:v for k,v in topics.items() if v < request.threshold}

                isDetection = False
                violationDescription = []
                if request.allowList:
                    isDetection = not any([allowedTopic in matched_topics.keys() for allowedTopic in request.allowList])
                    if isDetection:
                        violationDescription.append("Text matched none of the allowed topics: {}".format(request.allowList))

                if request.blockList:
                    blockMatches = [blockedTopic for blockedTopic in request.blockList if blockedTopic in matched_topics.keys()]
                    if blockMatches:
                        isDetection = True
                        violationDescription.append("Text matched the following blocked topic(s): {}".format(blockMatches))

                contents_analyses.append(
                    ContentAnalysisResponse(
                        start=0,
                        end=len(texts[idx]),
                        detection="mmluTopicMatch",
                        detection_type="mmluTopicMatch",
                        topics=matched_topics,
                        violation=isDetection,
                        violationDescription=violationDescription,
                        text=texts[idx],
                        evidences=[],
                    )
                )
        return contents_analyses

### local testing
if __name__ == "__main__":
    detector = Detector()
    request = ContentAnalysisHttpRequest(
        contents=["How far away is the Sun from the center of the Milky Way?", "What is the healthiest vegetable?", "What is the square root of 256?"],
        allowList=['astronomy'],
        blockList=['nutrition'],
    )

    analyses = detector.run(request)
    print(analyses)