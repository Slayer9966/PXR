from pydantic import BaseModel
from datetime import datetime

class ObjFileCreate(BaseModel):
    user_id: int

class ObjFileResponse(BaseModel):
    id: int
    user_id: str
    filepath: str
    created_at: datetime

    class Config:
     from_attributes = True
