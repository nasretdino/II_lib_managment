from typing import Annotated

from fastapi import APIRouter, Depends

from src.core import get_logger

from .dependencies import get_agents_service
from .schemas import AgentRunRequest, AgentRunResponse
from .services import AgentsService


logger = get_logger(module="agents", component="router")


router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/run", response_model=AgentRunResponse)
async def run_agents_pipeline(
    body: AgentRunRequest,
    service: Annotated[AgentsService, Depends(get_agents_service)],
):
    logger.info(
        "POST /agents/run: question_len={}, history_items={}, max_iterations={}",
        len(body.question),
        len(body.conversation_history),
        body.max_iterations,
    )
    result = await service.run(
        question=body.question,
        conversation_history=body.conversation_history,
        max_iterations=body.max_iterations,
    )
    logger.info(
        "POST /agents/run completed: query_type={}, iterations={}, approved={}",
        result.query_type,
        result.iterations,
        result.is_approved,
    )
    return result
