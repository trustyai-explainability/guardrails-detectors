from enum import Enum
from typing import List, Optional

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


class ContentAnalysisResponse(BaseModel):
    start: int = Field(example=14)
    end: int = Field(example=26)
    detection: str = Field(example="huggingface_model")
    detection_type: str = Field(example="Sequence Classification")
    score: float = Field(example=0.8)
    sequence_classification: Optional[str] = Field(example="Sequence Classification")
    sequence_probability: Optional[float] = Field(example=0.8)
    token_classifications: Optional[List[str]] = Field(example="[0, 1, 0")
    token_probabilities: Optional[List[float]] = Field(example="[0.5, 0.8, 0.0]")
    text: str = Field(example="A problematic word is fiddlesticks")
    evidences: Optional[List[EvidenceObj]] = Field(
        description="Optional field providing evidences for the provided detection",
        default=[],
    )


class ContentsAnalysisResponse(RootModel):
    root: List[List[ContentAnalysisResponse]] = Field(
        title="Response Text Content Analysis Unary Handler Api V1 Text Content Post"
    )


class Error(BaseModel):
    code: int
    message: str
