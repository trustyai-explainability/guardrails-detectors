from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, RootModel


class Evidence(BaseModel):
    source: str = Field(
        title="Source",
        example="https://en.wikipedia.org/wiki/IBM",
        description="Source of the evidence, it can be url of the evidence etc",
    )


class EvidenceType(str, Enum):
    url = "url"
    title = "title"


class EvidenceObj(BaseModel):
    type: EvidenceType = Field(
        title="EvidenceType",
        example="url",
        description="Type field signifying the type of evidence provided. Example url, title etc",
    )
    evidence: Evidence = Field(
        description="Evidence object, currently only containing source, but in future can contain other optional arguments like id, etc",
    )


class ContentAnalysisHttpRequest(BaseModel):
    contents: List[str] = Field(
        min_length=1,
        title="Contents",
        description="Field allowing users to provide list of texts for analysis. Note, results of this endpoint will contain analysis / detection of each of the provided text in the order they are present in the contents object.",
        example=[
            "Martians are like crocodiles; the more you give them meat, the more they want"
        ],
    )
    detector_params: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Detector parameters for evaluation (e.g., metric, criteria, etc.)",
        example={"metric": "safety"}
    )


class ContentAnalysisResponse(BaseModel):
    start: int = Field(example=0)
    end: int = Field(example=75)
    text: str = Field(example="This is a safe and helpful response")
    detection: str = Field(example="vllm_model")
    detection_type: str = Field(example="llm_judge")
    score: float = Field(example=0.8)
    evidences: Optional[List[EvidenceObj]] = Field(
        description="Optional field providing evidences for the provided detection",
        default=[],
    )
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata from evaluation")


class ContentsAnalysisResponse(RootModel):
    root: List[List[ContentAnalysisResponse]] = Field(
        title="Response Text Content Analysis LLM Judge"
    )


class Error(BaseModel):
    code: int
    message: str


class MetricsListResponse(BaseModel):
    """Response for listing available metrics."""
    metrics: List[str] = Field(description="List of available metric names")
    total: int = Field(description="Total number of available metrics")

class GenerationAnalysisHttpRequest(BaseModel):
    prompt: str = Field(description="Prompt is the user input to the LLM", example="What do you think about the future of AI?")
    generated_text: str = Field(description="Generated response from the LLM", example="The future of AI is bright but we need to be careful about the risks.")
    detector_params: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Detector parameters for evaluation (e.g., metric, criteria, etc.)",
        example={"metric": "safety"}
    )

class GenerationAnalysisResponse(BaseModel):
    detection: str = Field(example="safe")
    detection_type: str = Field(example="llm_judge")
    score: float = Field(example=0.8)
    evidences: Optional[List[EvidenceObj]] = Field(
        description="Optional field providing evidences for the provided detection",
        default=[],
    )
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata from evaluation")