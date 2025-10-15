import json
import logging

from fastapi import HTTPException

import jsonschema
import xml.etree.ElementTree as ET
import xmlschema
import yaml

from typing import List, Optional

from base_detector_registry import BaseDetectorRegistry
from detectors.common.scheme import ContentAnalysisResponse

logger = logging.getLogger(__name__)

def is_valid_json(s: str) -> Optional[ContentAnalysisResponse]:
    """Detect if the text contents is not valid JSON"""
    try:
        json.loads(s)
        return None
    except (ValueError, TypeError):
        return ContentAnalysisResponse(
            start=0,
            end=len(s),
            text=s,
            detection="invalid_json",
            detection_type= "file_type",
            score=1.0
        )


def is_valid_json_schema(s: str, schema: str) -> Optional[ContentAnalysisResponse]:
    """Detect if the text contents does not satisfy a provided JSON schema. To specify a schema, replace $SCHEMA with a JSON schema."""
    is_valid = is_valid_json(s)
    if is_valid is None:
        msg_data = json.loads(s)
    else:
        return is_valid


    # validate that the schema is valid json
    try:
        schema_data = json.loads(schema)
    except (ValueError, TypeError):
        return ContentAnalysisResponse(
            start=0,
            end=len(schema),
            text=s,
            detection="invalid_schema",
            detection_type="file_type",
            score=1.0
        )

    # validate the schema against the message
    try:
        jsonschema.validate(instance=msg_data, schema=schema_data)
        return None
    except jsonschema.ValidationError as e:
        return ContentAnalysisResponse(
            start=0,
            end=len(s),
            text=s,
            detection="json_schema_mismatch",
            detection_type="file_type",
            score=1.0
        )


def is_valid_yaml(s: str) -> Optional[ContentAnalysisResponse]:
    """Detect if the text contents is not valid YAML"""
    try:
        yaml.safe_load(s)
        return None
    except Exception:
        return ContentAnalysisResponse(
            start=0,
            end=len(s),
            text=s,
            detection="invalid_yaml",
            detection_type="file_type",
            score=1.0,
        )

def is_valid_yaml_schema(s: str, schema) -> Optional[ContentAnalysisResponse]:
    """Detect if the text contents does not satisfy a provided schema. To specify a schema, replace $SCHEMA with a JSON schema. That's not a typo, you validate YAML with a JSON schema!"""
    is_valid = is_valid_yaml(s)
    if is_valid is None:
        msg_data = yaml.safe_load(s)
    else:
        return is_valid

    # validate that the schema is valid json
    try:
        schema_data = json.loads(schema)
    except (ValueError, TypeError):
        return ContentAnalysisResponse(
            start=0,
            end=len(schema),
            text=s,
            detection="invalid_schema",
            detection_type="file_type",
            score=1.0
        )

    # validate the schema against the message
    try:
        jsonschema.validate(instance=msg_data, schema=schema_data)
        return None
    except jsonschema.ValidationError as e:
        return ContentAnalysisResponse(
            start=0,
            end=len(s),
            text=s,
            detection="yaml_schema_mismatch",
            detection_type="file_type",
            score=1.0
        )


def is_valid_xml(s: str) -> Optional[ContentAnalysisResponse]:
    """Detect if the text contents is not valid XML"""
    try:
        ET.fromstring(s)
        return None
    except Exception:
        return ContentAnalysisResponse(
            start=0,
            end=len(s),
            text=s,
            detection="invalid_xml",
            detection_type="file_type",
            score=1.0,
        )


def is_valid_xml_schema(s: str, schema) -> Optional[ContentAnalysisResponse]:
    """Detect if the text contents does not satisfy a provided XML schema. To specify a schema, replace $SCHEMA with an XML Schema Definition (XSD)"""
    is_valid = is_valid_xml(s)
    if is_valid is not None:
        return is_valid
    try:
        # schema is expected to be a string containing the XSD
        xs = xmlschema.XMLSchema(schema)
    except Exception:
        return ContentAnalysisResponse(
            start=0,
            end=len(schema),
            text=s,
            detection="invalid_xml_schema",
            detection_type="file_type",
            score=1.0
        )

    try:
        xs.validate(s)
        return None
    except xmlschema.XMLSchemaValidationError:
        return ContentAnalysisResponse(
            start=0,
            end=len(s),
            text=s,
            detection="xml_schema_mismatch",
            detection_type="file_type",
            score=1.0
        )



class FileTypeDetectorRegistry(BaseDetectorRegistry):
    def __init__(self):
        super().__init__("file_type")
        self.registry = {
            "json": is_valid_json,
            "xml": is_valid_xml,
            "yaml": is_valid_yaml,
            "json-with-schema:$SCHEMA": is_valid_json_schema,
            "xml-with-schema:$SCHEMA": is_valid_xml_schema,
            "yaml-with-schema:$SCHEMA": is_valid_yaml_schema,
        }

    def handle_request(self, content: str, detector_params: dict, headers: dict) -> List[ContentAnalysisResponse]:
        detections = []
        for file_type in self.get_detection_functions_from_params(detector_params):
            file_type_valid, func_name = True, None
            try:
                if file_type.startswith("json-with-schema"):
                    func_name = "json-with-schema"  # don't publish full schema to prometheus labels, to limit metric cardinality
                    with self.instrument_runtime(func_name):
                        result = is_valid_json_schema(content, file_type.split("json-with-schema:")[1])
                elif file_type.startswith("yaml-with-schema"):
                    func_name = "yaml-with-schema"  # don't publish full schema to prometheus labels, to limit metric cardinality
                    with self.instrument_runtime(func_name):
                        result = is_valid_yaml_schema(content, file_type.split("yaml-with-schema:")[1])
                elif file_type.startswith("xml-with-schema"):
                    func_name = "xml-with-schema" # as above
                    with self.instrument_runtime(func_name):
                        result = is_valid_xml_schema(content, file_type.split("xml-with-schema:")[1])
                elif file_type in self.registry:
                    func_name = file_type
                    with self.instrument_runtime(func_name):
                        result = self.registry[file_type](content)
                else:
                    func_name = "invalid_file_type"
                    file_type_valid = False
            except Exception as e:
                self.throw_internal_detector_error(func_name, logger, e, increment_requests=True)

            if not file_type_valid:
                raise HTTPException(status_code=400, detail=f"Unrecognized file type: {file_type}")

            # report results
            is_detection = result is not None
            self.increment_detector_instruments(func_name, is_detection)
            if is_detection:
                detections += [result]
        return detections
