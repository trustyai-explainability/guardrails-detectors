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
        example=["My email address is xx@domain.com and zzz@hotdomain.co.uk"],
    )
    regex_pattern: str = Field(
        title="Regex",
        description="Regex to be used for matching",
        example="[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}",
    )


class ContentAnalysisResponse(BaseModel):
    start: int = Field(example=14)
    end: int = Field(example=26)
    detection: str = Field(example="has_regex_match")
    detection_type: str = Field(example="regex_match")
    detection_value: bool = Field(example=True)
    text: str = Field(
        example="My email address is xx@domain.com and zzz@hotdomain.co.uk"
    )
    matches: List[str] = Field(
        description="List of matches found in the text",
        example=["xx@domain.com", "zzz@hotdomain.co.uk"],
    )
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
