## Built-In Detector Examples

### File Type Validation Example
```bash
curl -X POST http://localhost:8080/api/v1/text/contents \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      "{\"hello\": \"message\"}",
      "not valid json"
    ],
    "detector_params": {
      "file_type": [
        "json"
      ]
    }
  }'
```
Response:
```json
[
  [],
  [
    {
      "start": 0,
      "end": 14,
      "text": "not valid json",
      "detection": "invalid_json",
      "detection_type": "file_type",
      "score": 1.0,
      "evidences": null
    }
  ]
]
```

### PII Validation Example
```bash
curl -X POST http://localhost:8080/api/v1/text/contents \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      "Hi my email is abc@def.com",
      "There is a party@my house and you can reach me at 123-456-7890"
    ],
    "detector_params": {
      "regex": [
        "email", "us-phone-number"
      ]
    }
  }' | jq
```
Response:
```json
[
  [
    {
      "start": 15,
      "end": 26,
      "text": "abc@def.com",
      "detection": "email_address",
      "detection_type": "pii",
      "score": 1.0,
      "evidences": null
    }
  ],
  [
    {
      "start": 50,
      "end": 62,
      "text": "123-456-7890",
      "detection": "us-phone-number",
      "detection_type": "pii",
      "score": 1.0,
      "evidences": null
    }
  ]
]
```

### Get list of built-in detection algorithms:
```bash
curl http://localhost:8080/registry | jq
```
Response:
```json
{
  "regex": {
    "credit-card": "Detect credit cards in the text contents (Visa, MasterCard, Amex, Discover, Diners Club, JCB) with Luhn check",
    "email": "Detect email addresses in the text contents",
    "ipv4": "Detect IPv4 addresses in the text contents",
    "ipv6": "Detect IPv6 addresses in the text contents",
    "us-phone-number": "Detect US phone numbers in the text contents",
    "us-social-security-number": "Detect social security numbers in the text contents",
    "uk-post-code": "Detect UK post codes in the text contents",
    "$CUSTOM_REGEX": "Replace $CUSTOM_REGEX with a custom regex to define your own regex detector"
  },
  "file_type": {
    "json": "Detect if the text contents is not valid JSON",
    "xml": "Detect if the text contents is not valid XML",
    "yaml": "Detect if the text contents is not valid YAML",
    "json-with-schema:$SCHEMA": "Detect if the text contents does not satisfy a provided JSON schema. To specify a schema, replace $SCHEMA with a JSON schema.",
    "xml-with-schema:$SCHEMA": "Detect if the text contents does not satisfy a provided XML schema. To specify a schema, replace $SCHEMA with an XML Schema Definition (XSD)",
    "yaml-with-schema:$SCHEMA": "Detect if the text contents does not satisfy a provided schema. To specify a schema, replace $SCHEMA with a JSON schema. That's not a typo, you validate YAML with a JSON schema!"
  }
}
```