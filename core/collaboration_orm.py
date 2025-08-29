from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, JSON, Text, Float, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from core.db import Base

class CollaborationRoom(Base):
    __tablename__ = 'collaboration_rooms'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    creator_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))
    room_code = Column(String(20), unique=True, nullable=False)
    max_participants = Column(Integer, default=10)
    is_active = Column(Boolean, default=True)
    room_type = Column(String(50), default='study_group')
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    participants = relationship('RoomParticipant', back_populates='room')

class RoomParticipant(Base):
    __tablename__ = 'room_participants'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    room_id = Column(UUID(as_uuid=True), ForeignKey('collaboration_rooms.id', ondelete='CASCADE'))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))
    role = Column(String(20), default='participant')
    joined_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    room = relationship('CollaborationRoom', back_populates='participants')
    __table_args__ = (UniqueConstraint('room_id', 'user_id', name='_room_participant_uc'),)
