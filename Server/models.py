from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.sql import func
from database import Base

class ObjFile(Base):
    __tablename__ = "objfiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # assuming user_id is integer
    filepath = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
