import pytest
from fastapi.testclient import TestClient
import json


class TestFileTypeDetectors:
    @pytest.fixture
    def client(self):
        from detectors.built_in.app import app
        from detectors.built_in.file_type_detectors import FileTypeDetectorRegistry

        app.set_detector(FileTypeDetectorRegistry(), "file_type")

        return TestClient(app)

    @pytest.fixture
    def jsonschema(self):
        return json.dumps({
            "type": "object",
            "properties": {
                "a": {"type": "integer"}
            },
            "required": ["a"]
        })

    @pytest.fixture
    def xmlschema(self):
        return """<?xml version="1.0"?>
        <xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
          <xs:element name="root">
            <xs:complexType>
              <xs:sequence>
                <xs:element name="child" type="xs:string"/>
              </xs:sequence>
            </xs:complexType>
          </xs:element>
        </xs:schema>
        """


    # === JSON =====================================================================================
    def test_detect_content_valid_json(self, client: TestClient):
        payload = {
            "contents": ['{"a": 1, "b": 2}'],
            "detector_params": {"file_type": ["json"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        # Should be empty list for valid JSON
        assert resp.json()[0] == []

    def test_detect_content_invalid_json(self, client: TestClient):
        payload = {
            "contents": ['{a: 1, b: 2}'],
            "detector_params": {"file_type": ["json"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        detections = resp.json()[0]
        assert detections[0]["detection"] == "invalid_json"


    # === JSON SCHEMA ==============================================================================
    def test_json_schema_valid(self, client: TestClient, jsonschema):
        payload = {
            "contents": [json.dumps({"a": 1})],
            "detector_params": {"file_type": [f"json-with-schema:{jsonschema}"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        # Should be empty list for valid JSON and schema
        assert resp.json()[0] == []

    def test_json_schema_invalid(self, client: TestClient, jsonschema):
        payload = {
            "contents": [json.dumps({"a": "not_an_int"})],
            "detector_params": {"file_type": [f"json-with-schema:{jsonschema}"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        detections = resp.json()[0]
        assert detections[0]["detection"] == "json_schema_mismatch"

    def test_json_schema_invalid_json(self, client: TestClient, jsonschema):
        payload = {
            "contents": ['{a: 1}'],
            "detector_params": {"file_type": [f"json-with-schema:{jsonschema}"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        detections = resp.json()[0]
        assert detections[0]["detection"] == "invalid_json"


    def test_json_schema_invalid_json_schema(self, client: TestClient):
        # The schema expects an object with a required integer property "a"
        invalid_schema = '{"notvalidjson": {'
        payload = {
            "contents": [json.dumps({"a": 1})],
            "detector_params": {"file_type": [f"json-with-schema:{invalid_schema}"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        detections = resp.json()[0]
        assert detections[0]["detection"] == "invalid_schema"


    # === YAML =====================================================================================
    def test_detect_content_valid_yaml(self, client: TestClient):
        payload = {
            "contents": ['a: 1\nb: 2'],
            "detector_params": {"file_type": ["yaml"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        # Should be empty list for valid YAML
        assert resp.json()[0] == []

    def test_detect_content_invalid_yaml(self, client: TestClient):
        payload = {
            "contents": ['a: 1\nb: ['],
            "detector_params": {"file_type": ["yaml"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        detections = resp.json()[0]
        assert detections[0]["detection"] == "invalid_yaml"


    # === YAML SCHEMA ==============================================================================
    def test_detect_content_valid_yaml_schema(self, client: TestClient, jsonschema):
        payload = {
            "contents": ['a: 1\nb: 2'],
            "detector_params": {"file_type": [f"yaml-with-schema:{jsonschema}"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        # Should be empty list for valid YAML
        assert resp.json()[0] == []

    def test_detect_content_invalid_yaml_schema(self, client: TestClient, jsonschema):
        payload = {
            "contents": ['a: not_integer\nb: 2'],
            "detector_params": {"file_type": [f"yaml-with-schema:{jsonschema}"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        # Should be empty list for valid YAML
        assert resp.json()[0][0]['detection'] == "yaml_schema_mismatch"


    # === XML ======================================================================================
    def test_detect_content_valid_xml(self, client: TestClient):
        payload = {
            "contents": ['<root><child>data</child></root>'],
            "detector_params": {"file_type": ["xml"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        # Should be empty list for valid XML
        assert resp.json()[0] == []

    def test_detect_content_invalid_xml(self, client: TestClient):
        payload = {
            "contents": ['<root><child>data</root>'],
            "detector_params": {"file_type": ["xml"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        detections = resp.json()[0]
        assert detections[0]["detection"] == "invalid_xml"

    # === XML SCHEMA ===============================================================================
    def test_xml_schema_valid(self, client: TestClient, xmlschema):
        valid_xml = "<root><child>data</child></root>"

        # Valid XML and schema
        payload = {
            "contents": [valid_xml],
            "detector_params": {"file_type": [f"xml-with-schema:{xmlschema}"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        assert resp.json()[0] == []

    def test_xml_schema_validation_invalid_xml(self, client: TestClient, xmlschema):
        # Invalid XML (malformed)
        invalid_xml = "<root><child></root>"
        payload = {
            "contents": [invalid_xml],
            "detector_params": {"file_type": [f"xml-with-schema:{xmlschema}"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        detections = resp.json()[0]
        assert any(d["detection"] == "invalid_xml" for d in detections)

    def test_xml_schema_mismatch(self, client: TestClient, xmlschema):
        # Valid XML, but does not match schema (missing <child>)
        not_matching_xml = "<root></root>"
        payload = {
            "contents": [not_matching_xml],
            "detector_params": {"file_type": [f"xml-with-schema:{xmlschema}"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        detections = resp.json()[0]
        assert any(d["detection"] == "xml_schema_mismatch" for d in detections)

    # === MISC =====================================================================================
    def test_detect_content_unrecognized_filetype(self, client: TestClient):
        payload = {
            "contents": ['foo'],
            "detector_params": {"file_type": ["not_a_type"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 400
        data = resp.json()
        assert "message" in data
        assert "Unrecognized file type" in data["message"]

    def test_detect_content_single_filetype(self, client: TestClient):
        payload = {
            "contents": ['{a: 1, b: 2}'],
            "detector_params": {"file_type": "json"}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        detections = resp.json()[0]
        assert detections[0]["detection"] == "invalid_json"


    def test_multiple_filetype_valid_and_invalid(self, client: TestClient):
        import json
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "integer"}
            },
            "required": ["a"]
        }
        payload = {
            "contents": [
                '{"a": 1}',  # valid json, valid schema
                '{"a": "not_an_int"}',  # valid json, invalid schema
                '{a: 1}',  # invalid json
                '<root></root>',  # valid xml
                '<root><child></root>',  # invalid xml
                'a: 1\nb: 2',  # valid yaml
                'a: [',  # invalid yaml
            ],
            "detector_params": {
                "file_type": [
                    "json",
                    f"json-with-schema:{json.dumps(schema)}",
                    "xml",
                    "yaml"
                ]
            }
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        detections = resp.json()

        # 1: valid json, valid schema, valid xml/yaml
        assert not any(d["detection"] == "invalid_json" for d in detections[0])
        assert not any(d["detection"] == "json_schema_mismatch" for d in detections[0])

        # # 2: valid json, invalid schema
        # 1: valid json, valid schema, valid xml/yaml
        assert not any(d["detection"] == "invalid_json" for d in detections[1])
        assert any(d["detection"] == "json_schema_mismatch" for d in detections[1])

        # 3: invalid json
        assert len([d for d in detections[2] if d["detection"] == "invalid_json"]) == 2

        # 4: valid xml
        assert not any(d["detection"] == "invalid_xml" for d in detections[3])

        # 5: invalid xml
        assert any(d["detection"] == "invalid_xml" for d in detections[4])

        # 6: valid yaml
        assert not any(d["detection"] == "invalid_yaml" for d in detections[5])

        # 7: invalid yaml
        assert any(d["detection"] == "invalid_yaml" for d in detections[6])
    
    
    # === ERROR HANDLING & INVALID DETECTOR TYPES =================================================
    def test_unregistered_detector_kind_ignored(self, client: TestClient):
        """Test that requesting an unregistered detector kind returns 400 error"""
        payload = {
            "contents": ['{"a": 1}'],
            "detector_params": {"nonexistent_detector": ["some_value"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        # Should return 400 error since nonexistent_detector is not registered
        assert resp.status_code == 400

    def test_mixed_valid_invalid_detector_kinds(self, client: TestClient):
        """Test mixing valid and invalid detector kinds returns 400 error"""
        payload = {
            "contents": ['{a: 1, b: 2}'],
            "detector_params": {
                "file_type": ["json"],  # valid detector kind
                "nonexistent_detector": ["some_value"]  # invalid detector kind
            }
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        # Should return 400 error for the unregistered detector
        assert resp.status_code == 400
    
    def test_empty_detector_params(self, client: TestClient):
        """Test with empty detector_params"""
        payload = {
            "contents": ['{"a": 1}'],
            "detector_params": {}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 200
        # Should return empty list since no detectors are specified
        assert resp.json()[0] == []
    
    def test_multiple_invalid_file_types(self, client: TestClient):
        """Test multiple invalid file types to ensure all errors are handled"""
        payload = {
            "contents": ['test content'],
            "detector_params": {"file_type": ["invalid_type_1", "invalid_type_2"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 400
        data = resp.json()
        assert "message" in data
        assert "Unrecognized file type" in data["message"]

    def test_mixed_valid_invalid_file_types(self, client: TestClient):
        """Test mixing valid and invalid file types"""
        payload = {
            "contents": ['{a: 1, b: 2}'],
            "detector_params": {"file_type": ["json", "invalid_type"]}
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 400
        data = resp.json()
        assert "message" in data
        assert "Unrecognized file type" in data["message"]
    
    def test_case_sensitivity_file_types(self, client: TestClient):
        """Test case sensitivity of file types"""
        payload = {
            "contents": ['{"a": 1}'],
            "detector_params": {"file_type": ["JSON"]}  # uppercase
        }
        resp = client.post("/api/v1/text/contents", json=payload)
        assert resp.status_code == 400
        data = resp.json()
        assert "message" in data
        assert "Unrecognized file type" in data["message"]