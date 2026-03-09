from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import StreamingResponse

from .dependencies import get_chat_service
from .schemas import ChatRequest, SessionRead, MessageRead
from .services import ChatService


router = APIRouter(prefix="/chat", tags=["chat"])


def _format_sse(event: str, data: str) -> str:
    if event == "keepalive":
        return ":\n\n"

    payload_lines = [f"event: {event}"]
    lines = data.splitlines() or [""]
    payload_lines.extend(f"data: {line}" for line in lines)
    return "\n".join(payload_lines) + "\n\n"


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
):
    session = await service.prepare_session(request)

    async def event_generator():
        async for event in service.stream_response(request=request, session_id=session.id):
            yield _format_sse(event.event, event.data)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions", response_model=list[SessionRead])
async def get_sessions(
    user_id: Annotated[int, Query(gt=0)],
    service: Annotated[ChatService, Depends(get_chat_service)],
):
    return await service.get_sessions(user_id=user_id)


@router.get("/sessions/{session_id}/messages", response_model=list[MessageRead])
async def get_messages(
    session_id: Annotated[int, Path(gt=0)],
    service: Annotated[ChatService, Depends(get_chat_service)],
):
    return await service.get_messages(session_id=session_id)
