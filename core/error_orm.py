from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, JSON, Text, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from core.db import Base

class CalculationError(Base):
    __tablename__ = 'calculation_errors'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'))
    calculation_id = Column(UUID(as_uuid=True), ForeignKey('calculations.id', ondelete='SET NULL'))
    error_type = Column(String(100))
    error_message = Column(Text)
    stack_trace = Column(Text)
    input_data = Column(JSON)
    system_info = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
