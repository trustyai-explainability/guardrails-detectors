## LLM Judge Detector Examples

The LLM Judge detector integrates [vLLM Judge](https://github.com/trustyai-explainability/vllm_judge) into the Guardrails Detector ecosystem, enabling any OpenAI API compatible LLM to assess content using flexible, natural-language criteria.

### Local Setup

1. **Start an OpenAI-compatible LLM server** (example with vLLM):
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 9090
```
Server accessible at `http://localhost:9090`

2. **Build the detector image**:
```bash
export LLM_Judge_IMAGE=llm_judge-detector:latest
podman build -f detectors/Dockerfile.judge -t $LLM_Judge_IMAGE detectors
```

3. **Run the detector container**:
```bash
podman run --rm -p 8000:8000 -e VLLM_BASE_URL=http://host.docker.internal:9090 $LLM_Judge_IMAGE
```

### Examples

#### Example 1: Pre-built Metrics

Instead of writing complex prompts let us use pre-defined prompts for common scenarios. vLLM-Judge library provides various builtin metrics (which are pre-defined prompts) including moderation (`safety`, `toxicity`, `bias`), general purpose (`helpfulness`, `clarity`, `conciseness`, `relevance`, `coherence`) and others (like code quality, content quality). Let's use one of the pre-defined metrics from vLLM-Judge, eg., `safety`.

The `safety` metric evaluates the content safety across multiple dimensions like physical harm, psycological harm, social harm and vulnerable groups.

**Request:**
```bash
curl -s -X POST \
  "http://localhost:8000/api/v1/text/contents" \
  -H 'accept: application/json' \
  -H 'detector-id: llm_judge' \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": ["You dotard, I really hate this stuff", "I simply love this stuff"],
    "detector_params": {"metric":"safety"}
  }' | jq
```

**Response:** (with Qwen2.5-7B-instruct)
```json
[
  [
    {
      "start": 0,
      "end": 36,
      "text": "You dotard, I really hate this stuff",
      "detection": "UNSAFE",
      "detection_type": "llm_judge",
      "score": 0.2,
      "evidences": [],
      "metadata": {
        "reasoning": "The content contains a derogatory term and expresses strong negative emotions, which could potentially cause psychological harm or social conflict. It does not pose physical or legal risks but the emotional tone could be harmful."
      }
    }
  ],
  [
    {
      "start": 0,
      "end": 24,
      "text": "I simply love this stuff",
      "detection": "SAFE",
      "detection_type": "llm_judge",
      "score": 0.8,
      "evidences": [],
      "metadata": {
        "reasoning": "The statement is a simple expression of personal preference and does not contain any instructions or content that could cause physical, psychological, social, or legal harm. It is safe for all audiences."
      }
    }
  ]
]
```

**Note:** For all built-in metrics, lower scores indicate negative classifications (UNSAFE, TOXIC, BIASED) and higher scores indicate positive classifications (SAFE, NON_TOXIC, UNBIASED). Model reasoning is accessible via response metadata.

#### Example 2: Custom Criteria

Create custom evaluation criteria using simple natural language:

**Request:**
```bash
curl -s -X POST \
  "http://localhost:8000/api/v1/text/contents" \
  -H 'accept: application/json' \
  -H 'detector-id: llm_judge' \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": [
        "Deep Learning models learn by adjusting weights through backpropagation.",
        "Quantum computing is not different compared to classical computing."
    ],
    "detector_params": {
        "criteria": "technical accuracy for graduate students"
    }
}' | jq
```

**Response:** (with Qwen2.5-7B-instruct)
```json
[
  [
    {
      "start": 0,
      "end": 72,
      "text": "Deep Learning models learn by adjusting weights through backpropagation.",
      "detection": "True",
      "detection_type": "llm_judge",
      "score": 1.0,
      "evidences": [],
      "metadata": {
        "reasoning": "The statement is technically accurate. Deep Learning models indeed learn by adjusting weights through the process of backpropagation, which is a standard and well-understood method in the field."
      }
    }
  ],
  [
    {
      "start": 0,
      "end": 67,
      "text": "Quantum computing is not different compared to classical computing.",
      "detection": "FAIL",
      "detection_type": "llm_judge",
      "score": 0.2,
      "evidences": [],
      "metadata": {
        "reasoning": "The statement is incorrect as quantum computing fundamentally differs from classical computing in terms of principles, algorithms, and potential applications."
      }
    }
  ]
]
```

We get pretty ok results where model uses positive label (like 'True') and higher scores (like 1.0) for positive instances i.e, that satisfy the criteria and similarly negative label ('FAIL') and lower score (0.2) for negative instances i.e, that does not satisfy the criteria.

But how to specifically say which labels to use and how to assign scores? This is where the `rubric` parameter comes in.

#### Example 3: Custom Labels and Scoring with Rubrics

Use the `rubric` parameter to specify consistent decision labels and scoring criteria:

**Request:**
```bash
curl -s -X POST \
  "http://localhost:8000/api/v1/text/contents" \
  -H 'accept: application/json' \
  -H 'detector-id: llm_judge' \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": [
"Deep Learning models learn by adjusting weights through backpropagation.",
"Quantum computing is not different compared to classical computing."
],
    "detector_params": {
"criteria": "technical accuracy for graduate students",
"rubric": "Assign lower scores for inaccurate content and higher scores for accurate ones. Also assign appropriate decision labels like 'ACCURATE', 'INACCURATE' and 'SOMEWHAT_ACCURATE'."
}
  }' | jq
```

**Response:** (with Qwen2.5-7B-instruct)
```json
[
  [
    {
      "start": 0,
      "end": 72,
      "text": "Deep Learning models learn by adjusting weights through backpropagation.",
      "detection": "ACCURATE",
      "detection_type": "llm_judge",
      "score": 1.0,
      "evidences": [],
      "metadata": {
        "reasoning": "The statement is technically accurate. Deep Learning models indeed learn by adjusting weights through the process of backpropagation, which is a standard method for training neural networks."
      }
    }
  ],
  [
    {
      "start": 0,
      "end": 67,
      "text": "Quantum computing is not different compared to classical computing.",
      "detection": "INACCURATE",
      "detection_type": "llm_judge",
      "score": 0.2,
      "evidences": [],
      "metadata": {
        "reasoning": "Quantum computing operates on fundamentally different principles compared to classical computing, such as superposition and entanglement, which are not present in classical models."
      }
    }
  ]
]
```

Note that instead of generic labels (like True/False or 'PASS'/'FAIL'), now we get meaningful labels according to our `rubric`. 

If you want to specify a detailed `rubric` to explain what certain score number mean, you can do that as well! Just pass a 'score -> description' mapping for the `rubric` parameter. Or if you want to change the scoring range from 0-1 to 0-10, you can do so by passing `scale: [0, 10]` in detector_params.

#### Example 4: Template Variables

Parameterize criteria using template variables for reusability:

**Request:**
```bash
curl -s -X POST \
  "http://localhost:8000/api/v1/text/contents" \
  -H 'accept: application/json' \
  -H 'detector-id: llm_judge' \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": [
        "Deep Learning models learn by adjusting weights through backpropagation.",
        "Quantum computing is not different compared to classical computing."
    ],
    "detector_params": {
        "criteria": "technical accuracy for {level} students",
        "template_vars": {"level": "graduate"},
        "rubric": "Assign lower scores for inaccurate content and higher scores for accurate ones. Also assign appropriate decision labels like 'ACCURATE', 'INACCURATE' and 'SOMEWHAT_ACCURATE'."
    }
  }' | jq
```

Similar response as above.


#### Example 5: Advanced Logic with Jinja Templating

Add conditional logic using Jinja templating:

**Request:**
```bash
curl -s -X POST \
  "http://localhost:8000/api/v1/text/contents" \
  -H 'accept: application/json' \
  -H 'detector-id: llm_judge' \
  -H 'Content-Type: application/json' \
  -d '{
    "contents": [
        "Deep Learning models learn by adjusting weights through backpropagation.",
        "Quantum computing is not different compared to classical computing."
    ],
    "detector_params": {
        "criteria": "Evaluate this content for {audience}.\n{% if technical_level == '\''advanced'\'' %}\nPay special attention to technical accuracy and depth.\n{% else %}\nFocus on clarity and accessibility.\n{% endif %}",
        "template_vars": {"audience": "graduate students", "technical_level": "advanced"},
        "rubric": "Assign lower scores for inaccurate content and higher scores for accurate ones. Also assign appropriate decision labels like 'ACCURATE', 'INACCURATE' and 'SOMEWHAT_ACCURATE'.",
        "template_engine":"jinja2"
    }
  }' | jq
```

Similar response as above.

### Parameter Reference

Below is the full list of parameters that can be passed to `detector_params` to fully customize and build advanced detection criteria for your guardrails:

- `criteria`: Detailed description of what to evaluate for
- `rubric`: Scoring instructions for evaluation, can be string or dict
- `scale`: Numeric scale for score [min, max]
- `input`: Extra input/question/prompt that the content is responding to
- `metric`: Pre-defined metric name. If provided along with other params, those param fields will take precedence over metric fields
- `template_vars`: Variable mapping to substitute in templates
- `template_engine`: Template engine to use ('format' or 'jinja2'), default is 'format'
- `system_prompt`: Custom system message to take full control of the evaluator LLM persona
- `examples`: Few-shot examples. List of JSON objects, each JSON represents an example and must contain `content`, `score`, and `reasoning` fields and `reasoning` fields

### Get list of pre-defined metric names:


```bash
curl http://localhost:8000/api/v1/metrics | jq
```
Response:
```json
{
  "metrics": [
    "accuracy",
    "agent_performance_template",
    "api_docs_template",
    "appropriate",
    "bias_detection",
    "clarity",
    "code_quality",
    "code_review_template",
    "code_security",
    "coherence",
    "conciseness",
    "creativity",
    "customer_service_template",
    "educational_content_template",
    "educational_value",
    "factual",
    "helpfulness",
    "legal_appropriateness",
    "llama_guard_3_safety",
    "medical_accuracy",
    "medical_info_template",
    "preference",
    "product_review_template",
    "professionalism",
    "rag_evaluation_template",
    "relevance",
    "safety",
    "summarization_quality",
    "toxicity",
    "translation_quality",
    "writing_quality_template"
  ],
  "total": 31
}
```