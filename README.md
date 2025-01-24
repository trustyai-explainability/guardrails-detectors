# Detectors for the FMS Guardrails Orchestrator

[FMS Guardrails Orchestrator](https://github.com/foundation-model-stack/fms-guardrails-orchestrator) is an open source project led by IBM which provides a server for invocation of detectors on text generation input and output, and standalone detections. 

This repository is intended to provide a collection of detectors that are supported by [the TrustyAI team](https://github.com/trustyai-explainability).

## Detectors

At the moment, the following detectors are supported:

- `huggingface` -- a generic detector class that is intended to be compatible with any [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForSequenceClassification) or a specific kind of [AutoModelForCausalLM](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM), namely [GraniteForCausalLM](https://github.com/ibm-granite/granite-guardian); this detector exposes `/api/v1/text/contents` and thus, could be configured to be a detector of type: `text_contents` within the FMS Guardrails Orchestrator framework. This detector is also intended to be deployed as a [KServe](https://github.com/kserve/kserve) inference service. 





