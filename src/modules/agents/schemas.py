from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    question: str = Field(min_length=1, description="User question")
    conversation_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Optional chat history for retrieval and analysis",
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Maximum ARCADE critique loops",
    )


class AgentEventRead(BaseModel):
    type: str
    data: str
    iteration: int


class AgentRunResponse(BaseModel):
    question: str
    query_type: str
    retrieval_mode: str
    final_answer: str
    sources: list[str]
    iterations: int
    is_approved: bool
    events: list[AgentEventRead]
