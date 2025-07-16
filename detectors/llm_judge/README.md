# LLM Judge Detector

The LLM Judge detector integrates the [vLLM Judge](https://github.com/trustyai-explainability/vllm_judge) into the Guardrails Detector ecosystem. Please refer [llm_judge_examples](docs/llm_judge_examples.md) for usage details.

```
oc apply -f deploy/servingruntime.yaml
oc apply -f deploy/isvc.yaml
```