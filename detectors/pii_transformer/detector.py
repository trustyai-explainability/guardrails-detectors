import os
import sys

sys.path.insert(0, os.path.abspath(".."))
# from common.scheme import TextDetectionHttpRequest, TextDetectionResponse

import torch.nn
from common.app import logger
from scheme import (
    ContentAnalysisHttpRequest,
    ContentAnalysisResponse,
    ContentsAnalysisResponse,
)

# Detector imports
from transformers import AutoTokenizer, AutoModelForTokenClassification


class Detector:
    def __init__(self):
        # initialize the detector
        model_files_path = os.environ.get("PII_MODEL_PATH")
        logger.info(model_files_path)
        # The tokenizer is going to be using the data on the CPU
        self.tokenizer = AutoTokenizer.from_pretrained(model_files_path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=model_files_path,
        )

        logger.info("torch.cuda".upper() + " " + str(torch.cuda.is_available()))

        self.cuda_device = None

        if torch.cuda.is_available():
            # transparently taking a cuda gpu for an actor
            self.cuda_device = torch.device("cuda")
            torch.cuda.empty_cache()
            self.model.to(self.cuda_device)
            # self.tokenizer.to(self.cuda_device)
            # AttributeError: 'RobertaTokenizerFast' object has no attribute 'to'
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            logger.info("cuda_device".upper() + " " + str(self.cuda_device))

    def run(self, input: ContentAnalysisHttpRequest) -> ContentsAnalysisResponse:
        # run the classification for each entry on contents array
        # logger.info(tokenizer_parameters)
        contents_analyses = []
        for text in input.contents:
            content_analyses = []
            tokenized = self.tokenizer(
                text,
                max_length=len(text),
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            if self.cuda_device:
                logger.info("adding tokenized to CUDA")
                # If we are using a GPU, the tokens need to be there.
                tokenized = tokenized.to(self.cuda_device)
                # print (tokenized)

            # A BatchEncoding includes 'data', 'encodings', 'is_fast', and 'n_sequences'.
            model_out = self.model(**tokenized)
            # logger.info(model_out)
            # return logits
            logits = model_out.logits
            # Get the class with the highest probability, and use the model’s id2label mapping to convert it to a text label list
            predictions = torch.argmax(logits, dim=2)
            predicted_token_class = [
                self.model.config.id2label[p] for p in predictions[0].tolist()
            ]
            # check if predicted token class list contains elements other than 'O', if yes, then it is a PII
            pii_indicator = any([True for p in predicted_token_class if p != "O"])

            # # A List[float] seems like a sensible way to return this
            # if hap_score >= input.parameters["threshold"]:
            content_analyses.append(
                ContentAnalysisResponse(
                    start=0,
                    end=len(text),
                    detection="has_pii",
                    detection_type="pii",
                    pii_check=pii_indicator,
                    text=text,
                    predicted_token_class=predicted_token_class,
                    evidences=[],
                )
            )
            contents_analyses.append(content_analyses)

        return contents_analyses
