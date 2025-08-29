class VerificationLog(Base):
    __tablename__ = "verification_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user_profiles.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    verification_type = Column(String, nullable=False)
    result = Column(String, nullable=False)
    details = Column(String, nullable=True)
    admin_name = Column(String, nullable=True)
    comment = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    face_match_score = Column(Float, nullable=True)

class AdminUser(Base):
    __tablename__ = "admin_users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    role = Column(String, default="reviewer")  # reviewer, admin, super_admin
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, JSON, Text, Float, UniqueConstraint, ARRAY, DECIMAL
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from core.db import Base

class User(Base):
    __tablename__ = 'users'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    subscription_type = Column(String(50), default='free')
    profile = relationship('UserProfile', uselist=False, back_populates='user')

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    email_verified = Column(Boolean, default=False)
    verification_token = Column(String, nullable=True)
    verification_status = Column(String, default="pending")
    preferred_notation = Column(String(20), default='standard')
    theme = Column(String(20), default='light')
    language = Column(String(10), default='en')
    timezone = Column(String(50), default='UTC')
    notification_preferences = Column(JSON, default=dict)
    privacy_settings = Column(JSON, default=dict)
    user = relationship('User', back_populates='profile')

# Add more ORM models as needed for calculations, sessions, achievements, etc.
