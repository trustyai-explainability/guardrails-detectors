from enum import Enum
from typing import List, Optional, Union
from pydantic import BaseModel, Field, RootModel, ConfigDict
from guardrails.classes.validation_outcome import ValidationOutcome


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
        example=["My email address is xx@domain.com and zzz@hotdomain.co.uk"],
    )


class ContentAnalysisResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    start: int = Field(example=14)
    end: int = Field(example=26)
    detection_type: str = Field(example="detection_type")
    text: str = Field(
        example="My email address is xx@domain.com and zzz@hotdomain.co.uk"
    )
    validation_result: Union[str, ValidationOutcome]
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
