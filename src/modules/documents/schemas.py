from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentUpload(BaseModel):
    user_id: int


class DocumentRead(BaseModel):
    id: int
    filename: str
    content_type: str
    file_size: int
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
