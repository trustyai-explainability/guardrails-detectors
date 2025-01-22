from enum import Enum
from typing import List, Optional, Dict

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
    threshold: float = Field(
        default=.75,
        description="Determines the maximum distance between a prompt embedding and the topic centroid to still be considered part of that topic. E.g., all prompts within $threshold of the `astronomy` centroid are given the 'astronomy' label. "
    )
    allowList: List[str]= Field(
        description="If an allowList is provided, only prompts that match at least one of the provided topics will be allowed.",
        example=['anatomy', 'global_facts', 'high_school_mathematics']
    )
    blockList: List[str]= Field(
        description="If a blockList is provided, only prompts that match none of the provided topics will be allowed.",
        example=['anatomy', 'global_facts', 'high_school_mathematics']
    )


class ContentAnalysisResponse(BaseModel):
    start: int = Field(example=14)
    end: int = Field(example=26)
    detection: str = Field(example="mmluTopicMatch")
    detection_type: str = Field(example="mmluTopicMatch")
    text: str = Field(example="My favourite dish is pierogi")
    topics: Dict[str, float] = Field(example=['nutrition'])
    violation: bool = Field(example=True)
    violationDescription: List[str] = Field(
        description="Description of any possible violations for allowed or blocked topics present in the text.",
        example=["Text matched none of the allowed topics: ['astronomy', 'anatomy']", "Text matched blocked topic(s): ['nutrition']"])
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
