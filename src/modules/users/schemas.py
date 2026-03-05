from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _strip_str(v: Any) -> Any:
    return v.strip() if isinstance(v, str) else v


StrStripped = Annotated[str, BeforeValidator(_strip_str)]


class UserBase(BaseModel):
    name: StrStripped = Field(..., min_length=1, max_length=255)


class UserCreate(UserBase):
    pass


class UserUpdate(BaseModel):
    name: StrStripped | None = Field(None, min_length=1, max_length=255)


class UserFilter(BaseModel):
    name: str | None = None


class UserRead(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
