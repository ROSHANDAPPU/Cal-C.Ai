from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, JSON, Text, Float, UniqueConstraint, ARRAY, DECIMAL
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from core.db import Base

class Achievement(Base):
    __tablename__ = 'achievements'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100))
    difficulty = Column(String(20))
    icon_url = Column(String(500))
    points = Column(Integer, default=0)
    requirements = Column(JSON)
    is_active = Column(Boolean, default=True)
    user_achievements = relationship('UserAchievement', back_populates='achievement')

class UserAchievement(Base):
    __tablename__ = 'user_achievements'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))
    achievement_id = Column(UUID(as_uuid=True), ForeignKey('achievements.id', ondelete='CASCADE'))
    earned_at = Column(DateTime, default=datetime.utcnow)
    progress = Column(JSON)
    achievement = relationship('Achievement', back_populates='user_achievements')
    __table_args__ = (UniqueConstraint('user_id', 'achievement_id', name='_user_achievement_uc'),)
