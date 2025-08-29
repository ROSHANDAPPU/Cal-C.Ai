from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, JSON, Text, Float, UniqueConstraint, DECIMAL
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from core.db import Base

class LearningAnalytics(Base):
    __tablename__ = 'learning_analytics'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))
    subject = Column(String(100))
    skill_level = Column(DECIMAL(4,2))
    problems_solved = Column(Integer, default=0)
    average_time_per_problem = Column(Integer)
    accuracy_rate = Column(DECIMAL(3,2))
    learning_velocity = Column(DECIMAL(5,2))
    last_updated = Column(DateTime, default=datetime.utcnow)
    prediction_data = Column(JSON)
    __table_args__ = (UniqueConstraint('user_id', 'subject', name='_user_subject_uc'),)
