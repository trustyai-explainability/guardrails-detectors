import os
import sys

sys.path.insert(0, os.path.abspath(".."))
# from common.scheme import TextDetectionHttpRequest, TextDetectionResponse
import math
import torch.nn
from common.app import logger
from scheme import (
    ContentAnalysisHttpRequest,
    ContentAnalysisResponse,
    ContentsAnalysisResponse,
)

# Detector imports
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)


class Detector:
    risk_names = [
        "harm",
        "social_bias",
        "jailbreak",
        "profanity",
        "unethical_behavior",
        "sexual_content",
        "violence",
    ]

    def __init__(self):
        # initialize the detector
        model_files_path = os.environ.get("MODEL_DIR")
        logger.info(f"Loading model from {model_files_path}")
        # The tokenizer is going to be using the data on the CPU
        self.tokenizer = AutoTokenizer.from_pretrained(model_files_path, use_fast=True)
        config = AutoConfig.from_pretrained(model_files_path)
        logger.info("Config: {}".format(config))

        self.is_token_classifier = False
        self.is_causal_lm = False
        self.is_sequence_classifier = False

        if any("ForTokenClassification" in arch for arch in config.architectures):
            self.is_token_classifier = True
            logger.error("Token classification models are not supported.")
            raise ValueError("Token classification models are not supported.")
        elif any("GraniteForCausalLM" in arch for arch in config.architectures):
            self.is_causal_lm = True
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_files_path,
            )
        elif any("ForSequenceClassification" in arch for arch in config.architectures):
            self.is_sequence_classifier = True
            self.model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=model_files_path,
            )
        else:
            logger.error("Unsupported model architecture.")
            raise ValueError("Unsupported model architecture.")

        logger.info("torch.cuda?".upper() + " " + str(torch.cuda.is_available()))

        try:
            self.model_name = self.model.config["model_type"]
        except Exception:
            if self.is_token_classifier:
                self.model_name = "token_classifier"
            elif self.is_causal_lm:
                self.model_name = "causal_lm"
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

    def parse_output(self, output, input_len, nlogprobs, safe_token, unsafe_token):
        label, prob_of_risk = None, None
        if nlogprobs > 0:
            list_index_logprobs_i = [
                torch.topk(token_i, k=nlogprobs, largest=True, sorted=True)
                for token_i in list(output.scores)[:-1]
            ]
            if list_index_logprobs_i is not None:
                prob = self.get_probabilities(
                    list_index_logprobs_i, safe_token, unsafe_token
                )
                prob_of_risk = prob[1]
        res = self.tokenizer.decode(
            output.sequences[:, input_len:][0], skip_special_tokens=True
        ).strip()
        if unsafe_token.lower() == res.lower():
            label = unsafe_token
        elif safe_token.lower() == res.lower():
            label = safe_token
        else:
            label = "failed"
        return label, prob_of_risk.item()

    def get_probabilities(self, logprobs, safe_token, unsafe_token):
        safe_token_prob = 1e-50
        unsafe_token_prob = 1e-50
        for gen_token_i in logprobs:
            for logprob, index in zip(
                gen_token_i.values.tolist()[0], gen_token_i.indices.tolist()[0]
            ):
                decoded_token = self.tokenizer.convert_ids_to_tokens(index)
                if decoded_token.strip().lower() == safe_token.lower():
                    safe_token_prob += math.exp(logprob)
                if decoded_token.strip().lower() == unsafe_token.lower():
                    unsafe_token_prob += math.exp(logprob)
        probabilities = torch.softmax(
            torch.tensor([math.log(safe_token_prob), math.log(unsafe_token_prob)]),
            dim=0,
        )
        return probabilities

    def run(self, input: ContentAnalysisHttpRequest) -> ContentsAnalysisResponse:
        # run the classification for each entry on contents array
        # logger.info(tokenizer_parameters)
        contents_analyses = []
        for text in input.contents:
            content_analyses = []
            if self.is_causal_lm:
                messages = [{"role": "user", "content": text}]
                for risk_name in self.risk_names:
                    guardian_config = {"risk_name": risk_name}
                    input_ids = self.tokenizer.apply_chat_template(
                        messages,
                        guardian_config=guardian_config,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(self.model.device)
                    input_len = input_ids.shape[1]
                    with torch.no_grad():
                        output = self.model.generate(
                            input_ids,
                            do_sample=False,
                            max_new_tokens=20,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                    nlogprobs = 20
                    safe_token = "No"
                    unsafe_token = "Yes"
                    label, prob_of_risk = self.parse_output(
                        output, input_len, nlogprobs, safe_token, unsafe_token
                    )
                    content_analyses.append(
                        ContentAnalysisResponse(
                            start=0,
                            end=len(text),
                            detection=self.model_name,
                            detection_type="causal_lm",
                            score=prob_of_risk,
                            sequence_classification=risk_name,
                            sequence_probability=prob_of_risk,
                            token_classifications=None,
                            token_probabilities=None,
                            text=text,
                            evidences=[],
                        )
                    )

            else:
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
                        prediction = (
                            torch.argmax(logits, dim=1).detach().numpy().tolist()[0]
                        )
                        prediction_labels = self.model.config.id2label[prediction]
                        probability = (
                            torch.softmax(logits, dim=1)
                            .detach()
                            .numpy()[:, 1]
                            .tolist()[0]
                        )

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
