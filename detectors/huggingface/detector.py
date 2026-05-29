import os
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Union

from detectors.common.instrumented_detector import InstrumentedDetector

import json
import math
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
)
from detectors.common.app import logger
from detectors.common.scheme import (
    ContentAnalysisHttpRequest,
    ContentAnalysisResponse,
    ContentsAnalysisResponse,
)
import gc


def _parse_threshold_env():
    """Parse THRESHOLD env var. Returns float or 0.5 as default."""
    raw = os.environ.get("THRESHOLD")
    if raw is not None:
        try:
            val = float(raw)
            if not (0.0 <= val <= 1.0):
                logger.warning(f"THRESHOLD env var {val} is outside [0, 1]. Using anyway.")
            logger.info(f"THRESHOLD env var: {val}")
            return val
        except ValueError:
            logger.warning(f"Could not parse THRESHOLD env var: {raw}. Defaulting to 0.5.")
    return 0.5


def _parse_label_thresholds_env():
    """Parse LABEL_THRESHOLDS env var. Returns dict mapping label names to float thresholds."""
    raw = os.environ.get("LABEL_THRESHOLDS")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and all(
                isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)
                for k, v in parsed.items()
            ):
                logger.info(f"LABEL_THRESHOLDS env var: {parsed}")
                return parsed
            else:
                logger.warning(
                    f"Invalid LABEL_THRESHOLDS structure: {parsed!r}. "
                    "Expected dict of str -> numeric. Defaulting to empty."
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Could not parse LABEL_THRESHOLDS env var: {e}. Defaulting to empty.")
    return {}


def _parse_max_length_env():
    """Parse MAX_LENGTH env var. Returns positive int or 512 as default."""
    raw = os.environ.get("MAX_LENGTH")
    if raw is not None:
        try:
            val = int(raw)
            if val <= 0:
                logger.warning(f"MAX_LENGTH must be positive, got {val}. Defaulting to 512.")
                return 512
            logger.info(f"MAX_LENGTH env var: {val}")
            return val
        except ValueError:
            logger.warning(f"Could not parse MAX_LENGTH env var: {raw}. Defaulting to 512.")
    return 512


def _parse_safe_labels_env(default=None):
    if default is None:
        default = [0]
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
            logger.warning(f"Could not parse SAFE_LABELS env var: {e}. Defaulting to {default}.")
            return default
    logger.info(f"SAFE_LABELS env var not set: defaulting to {default}.")
    return default


@dataclass(frozen=True)
class _ResolvedParams:
    """Validated, immutable bundle of per-request detector parameters."""
    threshold: float
    label_thresholds: Dict[str, float]
    safe_labels: FrozenSet[Union[str, int]]
    max_length: int


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
        self.default_threshold = _parse_threshold_env()
        self.default_label_thresholds = _parse_label_thresholds_env()
        self.default_max_length = _parse_max_length_env()

        model_files_path = os.environ.get("MODEL_DIR")
        if not model_files_path:
            raise ValueError("MODEL_DIR environment variable is not set.")

        logger.info(f"Loading model from {model_files_path}")

        self.initialize_model(model_files_path)

        # For token classifiers, re-parse with "O" as the default safe label
        # rather than index 0, since "O" can appear at any index depending
        if self.is_token_classifier:
            self.safe_labels = _parse_safe_labels_env(default=["O"])

        # Cache model max length for use in _resolve_params
        self._model_max_length = self._get_model_max_length()

        # Clamp env-level max_length to model's actual capacity
        if self._model_max_length and self.default_max_length > self._model_max_length:
            logger.warning(
                f"MAX_LENGTH env ({self.default_max_length}) exceeds model maximum ({self._model_max_length}). Clamping."
            )
            self.default_max_length = self._model_max_length

        self.initialize_device()

    def initialize_model(self, model_files_path):
        """
        Load and configure the model and tokenizer.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_files_path, use_fast=True)
        except (ValueError, OSError, ImportError) as e:
            logger.warning(f"Failed to load fast tokenizer: {e}. Falling back to slow tokenizer.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_files_path, use_fast=False)
        config = AutoConfig.from_pretrained(model_files_path)
        logger.info(f"Model Config: {config}")

        self.is_token_classifier = False
        self.is_causal_lm = False
        self.is_sequence_classifier = False

        if any("ForTokenClassification" in arch for arch in config.architectures):
            self.is_token_classifier = True
            if not getattr(self.tokenizer, "is_fast", False):
                raise ValueError(
                    "Token classification requires a fast tokenizer for "
                    "offset mapping support, but only a slow tokenizer is "
                    "available for this model."
                )
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_files_path
            )
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

    @staticmethod
    def _is_numeric(value):
        """Check if value is a real numeric type (excluding bool)."""
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _is_positive_int(value):
        """Check if value is a positive int (excluding bool)."""
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    def _get_model_max_length(self):
        """Return the model's real max length, or None if not meaningfully set."""
        model_max = getattr(self.tokenizer, "model_max_length", None)
        # Many tokenizers set model_max_length to a huge sentinel (e.g. int(1e30))
        # when no real limit is configured — treat these as unbounded
        if model_max and model_max < 1_000_000:
            return model_max
        return None

    def _resolve_params(
        self,
        detector_params: Optional[Dict],
        direct_threshold: Optional[float] = None,
    ) -> "_ResolvedParams":
        """Resolve all detector_params into a validated bundle."""
        detector_params = detector_params or {}

        # --- threshold ---
        if direct_threshold is not None and self._is_numeric(direct_threshold):
            threshold = float(direct_threshold)
            if not (0.0 <= threshold <= 1.0):
                logger.warning(f"Direct threshold {threshold} is outside [0, 1]. Using anyway.")
        elif direct_threshold is not None:
            logger.warning(f"Invalid direct threshold: {direct_threshold!r}. Using default {self.default_threshold}.")
            threshold = self.default_threshold
        else:
            raw_t = detector_params.get("threshold")
            if raw_t is None:
                threshold = self.default_threshold
            elif self._is_numeric(raw_t):
                threshold = float(raw_t)
                if not (0.0 <= threshold <= 1.0):
                    logger.warning(f"Threshold {threshold} in detector_params is outside [0, 1]. Using anyway.")
            else:
                logger.warning(f"Invalid threshold in detector_params: {raw_t!r}. Using default {self.default_threshold}.")
                threshold = self.default_threshold

        # --- label_thresholds ---
        label_thresholds = dict(self.default_label_thresholds)
        request_lt = detector_params.get("label_thresholds", {})
        if not isinstance(request_lt, dict):
            logger.warning(f"Invalid label_thresholds in detector_params: {request_lt!r}. Ignoring.")
        else:
            for k, v in request_lt.items():
                if isinstance(k, str) and self._is_numeric(v):
                    label_thresholds[k] = float(v)
                else:
                    logger.warning(f"Ignoring invalid label_threshold entry: {k!r}={v!r}.")

        # --- safe_labels ---
        raw_sl = detector_params.get("safe_labels", [])
        if isinstance(raw_sl, (str, int)) and not isinstance(raw_sl, bool):
            raw_sl = [raw_sl]
        if not isinstance(raw_sl, list):
            logger.warning(f"Invalid safe_labels in detector_params: {raw_sl!r}. Ignoring.")
            safe_labels = frozenset(self.safe_labels)
        else:
            valid = []
            for item in raw_sl:
                if isinstance(item, (str, int)) and not isinstance(item, bool):
                    valid.append(item)
                else:
                    logger.warning(f"Ignoring invalid safe_label entry: {item!r}.")
            safe_labels = frozenset(self.safe_labels) | frozenset(valid)

        # --- max_length ---
        raw_ml = detector_params.get("max_length")
        if raw_ml is None:
            max_length = self.default_max_length
        elif self._is_positive_int(raw_ml):
            max_length = raw_ml
        else:
            logger.warning(f"Invalid max_length in detector_params: {raw_ml!r}. Using default {self.default_max_length}.")
            max_length = self.default_max_length

        model_max = self._model_max_length
        if model_max and max_length > model_max:
            logger.warning(
                f"Requested max_length {max_length} exceeds model maximum {model_max}. Clamping to {model_max}."
            )
            max_length = model_max

        return _ResolvedParams(
            threshold=threshold,
            label_thresholds=label_thresholds,
            safe_labels=safe_labels,
            max_length=max_length,
        )

    def process_sequence_classification(self, text, detector_params=None, threshold=None):
        params = self._resolve_params(detector_params, direct_threshold=threshold)
        content_analyses = []
        tokenized = self.tokenizer(
            text,
            max_length=params.max_length,
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
                effective_threshold = params.label_thresholds.get(label, params.threshold)
                if (
                        prob >= effective_threshold
                        and idx not in params.safe_labels
                        and label not in params.safe_labels
                ):
                    detection_value = getattr(self.model.config, "problem_type", None)
                    content_analyses.append(
                        ContentAnalysisResponse(
                            start=0,
                            end=len(text),
                            detection_type=label,
                            score=float(prob),
                            text=text,
                            evidences=[],
                            **({"detection": detection_value} if detection_value is not None else {})
                        )
                    )
        return content_analyses

    def process_token_classification(self, text, detector_params=None, threshold=None):
        params = self._resolve_params(detector_params, direct_threshold=threshold)
        content_analyses = []
        tokenized = self.tokenizer(
            text,
            max_length=params.max_length,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_offsets_mapping=True,
        )
        # offset_mapping is not a model input — extract before sending to device
        offset_mapping = tokenized.pop("offset_mapping")[0]
        if self.cuda_device:
            tokenized = tokenized.to(self.cuda_device)

        with torch.no_grad():
            logits = self.model(**tokenized).logits
            probabilities = torch.softmax(logits, dim=2).detach().cpu().numpy()[0]

            # Per token, pick the single highest-probability non-safe label
            # above threshold. This avoids overlapping spans and preserves
            # left-to-right ordering.
            detected_tokens = []
            for token_idx, token_probs in enumerate(probabilities):
                char_start, char_end = offset_mapping[token_idx].tolist()
                # Skip special tokens (e.g. [CLS], [SEP]) which have (0, 0) offsets
                if char_start == 0 and char_end == 0:
                    continue

                best_label = None
                best_prob = -1.0
                for label_idx, prob in enumerate(token_probs):
                    label = self.model.config.id2label[label_idx]
                    prob = float(prob)
                    effective_threshold = params.label_thresholds.get(label, params.threshold)
                    if (
                        prob >= effective_threshold
                        and label_idx not in params.safe_labels
                        and label not in params.safe_labels
                        and prob > best_prob
                    ):
                        best_label = label
                        best_prob = prob

                if best_label is not None:
                    detected_tokens.append({
                        "token_idx": token_idx,
                        "char_start": int(char_start),
                        "char_end": int(char_end),
                        "label": best_label,
                        "prob": best_prob,
                    })

            # Group adjacent tokens with the same label into spans
            spans = []
            for token in detected_tokens:
                if (
                    spans
                    and spans[-1]["label"] == token["label"]
                    and token["token_idx"] == spans[-1]["last_token_idx"] + 1
                ):
                    spans[-1]["char_end"] = token["char_end"]
                    spans[-1]["last_token_idx"] = token["token_idx"]
                    spans[-1]["probs"].append(token["prob"])
                else:
                    spans.append({
                        "char_start": token["char_start"],
                        "char_end": token["char_end"],
                        "label": token["label"],
                        "last_token_idx": token["token_idx"],
                        "probs": [token["prob"]],
                    })

            detection_value = getattr(self.model.config, "problem_type", None)
            for span in spans:
                score = sum(span["probs"]) / len(span["probs"])
                content_analyses.append(
                    ContentAnalysisResponse(
                        start=span["char_start"],
                        end=span["char_end"],
                        detection_type=span["label"],
                        score=score,
                        text=text[span["char_start"]:span["char_end"]],
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
                elif self.is_token_classifier:
                    analyses = self.process_token_classification(
                        text, detector_params=getattr(input, "detector_params", None)
                    )
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
