# Detector Algorithms for the FMS Guardrails Orchestrator

[FMS Guardrails Orchestrator](https://github.com/foundation-model-stack/fms-guardrails-orchestrator) is an open source project led by IBM which provides a server for invocation of detectors on text generation input and output, and standalone detections. 

This repository is intended to provide a collection of detector algorithms and microservices that are supported by [the TrustyAI team](https://github.com/trustyai-explainability).

## Detectors

At the moment, the following detectors are supported:

- `huggingface` -- a generic detector class that is intended to be compatible with any [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForSequenceClassification) or a specific kind of [AutoModelForCausalLM](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM), namely [GraniteForCausalLM](https://github.com/ibm-granite/granite-guardian); this detector exposes `/api/v1/text/contents` and thus, could be configured to be a detector of type: `text_contents` within the FMS Guardrails Orchestrator framework. This detector is also intended to be deployed as a [KServe](https://github.com/kserve/kserve) inference service. 
- `llm_judge` -- Integrates the [vLLM Judge](https://github.com/trustyai-explainability/vllm_judge) library to use LLM-as-a-judge based guardrailing architecture
- `builtIn` -- Small, lightweight detection functions that are deployed out-of-the-box alongside the [Guardrails Orchestrator](https://github.com/foundation-model-stack/fms-guardrails-orchestrator). The built-in detectors provide a number of heuristic or algorithmic detection functions, such as:
  - Regex-based detections, with pre-written regexes for flagging various Personally Identifiable Information items like emails or phone numbers, as well as the ability to provide custom regexes
  - File-type validations, for verifying if model input/output is valid JSON, XML, or YAML


## Building

* `huggingface`: podman build -f src/guardrails_detectors/Dockerfile.hf detectors
* `llm_judge`: podman build -f src/guardrails_detectors/Dockerfile.llm_judge detectors
* `builtIn`: podman build -f src/guardrails_detectors/Dockerfile.builtIn detectors

## Running locally
* `builtIn`: podman run -p 8080:8080 $BUILT_IN_IMAGE

## Examples

- Check out [built-in detector examples](docs/builtin_examples.md) to see how to use the built-in detectors for file type validation and personally identifiable information (PII) detection
- Check out [Hugging Face detector examples](docs/hf_examples.md) to see how to use the Hugging Face detectors for detecting toxic content and prompt injection 

## API
See [IBM Detector API](https://foundation-model-stack.github.io/fms-guardrails-orchestrator/?urls.primaryName=Detector+API)

## License

This project is licensed under the Apache License Version 2.0 - see the [LICENSE](./LICENSE) file for details.
