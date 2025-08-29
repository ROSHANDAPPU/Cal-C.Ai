from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, JSON, Text, Float, UniqueConstraint, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from core.db import Base

class Formula(Base):
    __tablename__ = 'formulas'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    formula_latex = Column(Text, nullable=False)
    category = Column(String(100))
    subcategory = Column(String(100))
    description = Column(Text)
    variables = Column(JSON)
    constants = Column(JSON)
    difficulty_level = Column(Integer)
    usage_count = Column(Integer, default=0)
    tags = Column(ARRAY(Text))
    created_at = Column(DateTime, default=datetime.utcnow)
    user_formulas = relationship('UserFormula', back_populates='formula')

class UserFormula(Base):
    __tablename__ = 'user_formulas'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'))
    formula_id = Column(UUID(as_uuid=True), ForeignKey('formulas.id', ondelete='CASCADE'))
    custom_name = Column(String(255))
    custom_variables = Column(JSON)
    is_favorite = Column(Boolean, default=False)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    formula = relationship('Formula', back_populates='user_formulas')
    __table_args__ = (UniqueConstraint('user_id', 'formula_id', name='_user_formula_uc'),)
