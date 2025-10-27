import os

from detectors.common.instrumented_detector import InstrumentedDetector

import json
import math
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from detectors.common.app import logger
from detectors.common.scheme import (
    ContentAnalysisHttpRequest,
    ContentAnalysisResponse,
    ContentsAnalysisResponse,
)
import gc


def _parse_safe_labels_env():
    if os.environ.get("SAFE_LABELS"):
        try:
            parsed = json.loads(os.environ.get("SAFE_LABELS"))
            if isinstance(parsed, (int, str)):
                logger.info(f"SAFE_LABELS env var: {parsed}")
                return [parsed]
            if isinstance(parsed, list) and all(isinstance(x, (int, str)) for x in parsed):
                logger.info(f"SAFE_LABELS env var: {parsed}")
                return parsed
        except Exception as e:
            logger.warning(f"Could not parse SAFE_LABELS env var: {e}. Defaulting to [0].")
            return [0]
    logger.info("SAFE_LABELS env var not set: defaulting to [0].")
    return [0]


class Detector(InstrumentedDetector):
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
        """
        Initialize the Detector class by setting up the model, tokenizer, and device.
        """
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.cuda_device = None
        self.model_name = "unknown"
        self.safe_labels = _parse_safe_labels_env()

        model_files_path = os.environ.get("MODEL_DIR")
        if not model_files_path:
            raise ValueError("MODEL_DIR environment variable is not set.")

        logger.info(f"Loading model from {model_files_path}")

        self.initialize_model(model_files_path)
        self.initialize_device()

    def initialize_model(self, model_files_path):
        """
        Load and configure the model and tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_files_path, use_fast=True)
        config = AutoConfig.from_pretrained(model_files_path)
        logger.info(f"Model Config: {config}")

        self.is_token_classifier = False
        self.is_causal_lm = False
        self.is_sequence_classifier = False

        if any("ForTokenClassification" in arch for arch in config.architectures):
            self.is_token_classifier = True
            logger.error("Token classification models are not supported.")
            raise ValueError("Token classification models are not supported.")
        elif any("GraniteForCausalLM" in arch for arch in config.architectures):
            self.is_causal_lm = True
            self.model = AutoModelForCausalLM.from_pretrained(model_files_path)
        elif any("ForSequenceClassification" in arch for arch in config.architectures):
            self.is_sequence_classifier = True
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_files_path
            )
        else:
            logger.error("Unsupported model architecture.")
            raise ValueError("Unsupported model architecture.")

        if self.is_causal_lm:
            self.model_name = "causal_lm"
        elif self.is_sequence_classifier:
            self.model_name = "sequence_classifier"
        elif self.is_token_classifier:
            self.model_name = "token_classifier"
        else:
            self.model_name = "unknown"

        self.registry_name = self.model_name

        # set by k8s to be the pod name
        if os.environ.get("DETECTOR_NAME"):
            pod_name = os.environ.get("DETECTOR_NAME")
            if "-predictor" in pod_name:
                # recover the original ISVC name as specified by the user
                pod_name = pod_name.split("-predictor")[0]
            self.function_name = pod_name
        else:
            self.function_name = os.path.basename(model_files_path)

        logger.info(f"Model type detected: {self.model_name}")

    def initialize_device(self):
        """
        Set up the device for computation (CPU or CUDA).
        """
        if torch.cuda.is_available():
            try:
                self.cuda_device = torch.device("cuda")
                self.model.to(self.cuda_device)
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
                logger.info(f"CUDA device initialized: {self.cuda_device}")
            except RuntimeError as e:
                logger.warning(f"Failed to initialize CUDA device: {e}")
                self.cuda_device = None
        else:
            logger.info("CUDA is not available. Using CPU.")
            self.cuda_device = torch.device("cpu")

    def parse_output(self, output, input_len, nlogprobs, safe_token, unsafe_token):
        """
        Parse the model's output to determine the label and probability of risk.

        Args:
            output: Model output tensor.
            input_len: Length of the input sequence.
            nlogprobs: Number of log probabilities to consider.
            safe_token: Token representing a safe classification.
            unsafe_token: Token representing an unsafe classification.

        Returns:
            label (str): The classification label (e.g., safe, unsafe, or failed).
            prob_of_risk (float): Probability of risk.
        """
        label, prob_of_risk = None, None
        if nlogprobs > 0:
            list_index_logprobs_i = [
                torch.topk(token_i, k=nlogprobs, largest=True, sorted=True)
                for token_i in list(output.scores)[:-1]
            ]
            prob = self.get_probabilities(
                list_index_logprobs_i, safe_token, unsafe_token
            )
            prob_of_risk = prob[1]

        res = self.tokenizer.decode(
            output.sequences[:, input_len:][0], skip_special_tokens=True
        ).strip()
        label_mapping = {
            unsafe_token.lower(): unsafe_token,
            safe_token.lower(): safe_token,
        }
        label = label_mapping.get(res.lower(), "failed")
        return label, prob_of_risk.item() if prob_of_risk else None

    def get_probabilities(self, logprobs, safe_token, unsafe_token):
        """
        Calculate probabilities for safe and unsafe tokens based on log probabilities.

        Args:
            logprobs: Log probabilities from model output.
            safe_token: Token representing a safe classification.
            unsafe_token: Token representing an unsafe classification.

        Returns:
            probabilities (torch.Tensor): Tensor of probabilities for safe and unsafe tokens.
        """
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

    def process_causal_lm(self, text):
        messages = [{"role": "user", "content": text}]
        content_analyses = []
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
            label, prob_of_risk = self.parse_output(output, input_len, 20, "No", "Yes")
            content_analyses.append(
                ContentAnalysisResponse(
                    start=0,
                    end=len(text),
                    detection=self.model_name,
                    detection_type="causal_lm",
                    score=prob_of_risk,
                    text=text,
                    evidences=[],
                )
            )
        return content_analyses

    def process_sequence_classification(self, text, detector_params=None, threshold=None):
        detector_params = detector_params or {}
        if threshold is None:
            threshold = detector_params.get("threshold", 0.5)
        # Merge safe_labels from env and request
        request_safe_labels = set(detector_params.get("safe_labels", []))
        all_safe_labels = set(self.safe_labels) | request_safe_labels
        content_analyses = []
        tokenized = self.tokenizer(
            text,
            max_length=512,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        if self.cuda_device:
            tokenized = tokenized.to(self.cuda_device)

        with torch.no_grad():
            logits = self.model(**tokenized).logits
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
            for idx, prob in enumerate(probabilities):
                label = self.model.config.id2label[idx]
                # Exclude by index or label name
                if (
                        prob >= threshold
                        and idx not in all_safe_labels
                        and label not in all_safe_labels
                ):
                    detection_value = getattr(self.model.config, "problem_type", None)
                    content_analyses.append(
                        ContentAnalysisResponse(
                            start=0,
                            end=len(text),
                            detection_type=label,
                            score=prob,
                            text=text,
                            evidences=[],
                            **({"detection": detection_value} if detection_value is not None else {})
                        )
                    )
        return content_analyses

    def run(self, input: ContentAnalysisHttpRequest) -> ContentsAnalysisResponse:
        """
        Run the content analysis for each input text.

        Args:
            input (ContentAnalysisHttpRequest): Input request containing texts to analyze.

        Returns:
            ContentsAnalysisResponse: The aggregated response for all input texts.
        """
        contents_analyses = []
        with self.instrument_runtime(self.function_name):
            for text in input.contents:
                if self.is_causal_lm:
                    analyses = self.process_causal_lm(text)
                elif self.is_sequence_classifier:
                    analyses = self.process_sequence_classification(
                        text, detector_params=getattr(input, "detector_params", None)
                    )
                else:
                    raise ValueError("Unsupported model type for analysis.")
                contents_analyses.append(analyses)
        is_detection = any(len(analyses) > 0 for analyses in contents_analyses)
        self.increment_detector_instruments(self.function_name, is_detection=is_detection)
        return contents_analyses

    def close(self) -> None:
        """Clean up model and tokenizer resources."""
        if self.model:
            if hasattr(self.model, 'to') and hasattr(self.model, 'device') and self.model.device.type != "cpu":
                self.model = self.model.to(torch.device("cpu"))
            self.model = None

        if self.tokenizer:
            self.tokenizer = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
