from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, JSON, Text, Float, UniqueConstraint, ARRAY, DECIMAL
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from core.db import Base

class CalculationSession(Base):
    __tablename__ = 'calculation_sessions'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))
    session_name = Column(String(255))
    session_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_shared = Column(Boolean, default=False)
    share_token = Column(String(100), unique=True)
    tags = Column(ARRAY(Text))
    session_metadata = Column(JSON)
    calculations = relationship('Calculation', back_populates='session')

class Calculation(Base):
    __tablename__ = 'calculations'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('calculation_sessions.id', ondelete='CASCADE'))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))
    calculation_type = Column(String(100), nullable=False)
    input_data = Column(JSON, nullable=False)
    input_method = Column(String(50))
    processed_input = Column(JSON)
    calculation_steps = Column(JSON)
    result = Column(JSON, nullable=False)
    execution_time_ms = Column(Integer)
    ai_confidence_score = Column(DECIMAL(3,2))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_bookmarked = Column(Boolean, default=False)
    tags = Column(ARRAY(Text))
    notes = Column(Text)
    difficulty_level = Column(Integer)
    session = relationship('CalculationSession', back_populates='calculations')
