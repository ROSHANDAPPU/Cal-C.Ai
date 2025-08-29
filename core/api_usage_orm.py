from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, JSON, Text, Float, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from core.db import Base

class ApiUsage(Base):
    __tablename__ = 'api_usage'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))
    endpoint = Column(String(255))
    method = Column(String(10))
    calculations_count = Column(Integer, default=0)
    ai_requests_count = Column(Integer, default=0)
    reset_date = Column(DateTime, default=datetime.utcnow)
    monthly_limit = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint('user_id', 'endpoint', 'reset_date', name='_api_usage_uc'),)
