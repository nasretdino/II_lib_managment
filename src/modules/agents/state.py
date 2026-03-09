from typing import Any, Literal, TypedDict


QueryType = Literal["search", "analytics", "podcast"]


class AgentEvent(TypedDict):
    type: str
    data: str
    iteration: int


class AgentState(TypedDict, total=False):
    question: str
    conversation_history: list[dict[str, str]]

    query_type: QueryType
    retrieval_mode: str
    context: str
    sources: list[str]

    draft_answer: str
    critique: str
    correction: str

    is_approved: bool
    needs_more_context: bool
    citations_valid: bool

    iteration: int
    max_iterations: int

    route_after_decision: Literal["retry_retrieve", "retry_analyze", "end"]

    final_answer: str
    events: list[AgentEvent]
    raw: dict[str, Any]
