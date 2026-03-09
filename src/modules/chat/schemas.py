from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    session_id: int | None = Field(default=None, gt=0)
    user_id: int = Field(gt=0)
    message: str = Field(min_length=1)


class ChatEvent(BaseModel):
    event: str
    data: str = ""


class SessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    created_at: datetime
