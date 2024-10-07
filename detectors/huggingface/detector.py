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
from transformers import (AutoConfig, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification)


class Detector:
    def __init__(self):
        # initialize the detector
        model_files_path = os.environ.get("MODEL_DIR")
        logger.info(f"Loading model from {model_files_path}")
        # The tokenizer is going to be using the data on the CPU
        self.tokenizer = AutoTokenizer.from_pretrained(model_files_path, use_fast=True)
        config = AutoConfig.from_pretrained(model_files_path)
        logger.info("Config: {}".format(config))


        self.is_token_classifier = False

        if self.is_token_classifier:
            pass
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=model_files_path,
            )

        logger.info("torch.cuda?".upper() + " " + str(torch.cuda.is_available()))

        try:
            self.model_name = self.model.config["model_type"]
        except Exception:
            if self.is_token_classifier:
                self.model_name = "token_classifier"
            else:
                self.model_name = "sequence_classifier"
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
            with torch.no_grad():
                logger.info("tokens: {}".format(tokenized))
                model_out = self.model(**tokenized)
                # logger.info(model_out)
                # return logits
                logits = model_out.logits

                if self.is_token_classifier:
                    pass
                else:
                    # Get the class with the highest probability, and use the model’s id2label mapping to convert it to a text label list
                    prediction = torch.argmax(logits, dim=1).detach().numpy().tolist()[0]
                    prediction_labels = self.model.config.id2label[prediction]
                    probability = torch.softmax(logits, dim=1).detach().numpy()[:,1].tolist()[0]

                    content_analyses.append(
                        ContentAnalysisResponse(
                            start=0,
                            end=len(text),
                            detection=self.model_name,
                            detection_type="sequence_classification",
                            score=probability,
                            sequence_classification=prediction_labels,
                            sequence_probability=probability,
                            token_classifications=None,
                            token_probabilities=None,
                            text=text,
                            evidences=[],
                        )
                    )
            contents_analyses.append(content_analyses)

        return contents_analyses
