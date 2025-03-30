import os
import hashlib
import json
import datetime
from typing import List, Dict, Optional, Any
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    Text,
    DateTime,
    Boolean,
    JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from sqlalchemy.sql import func

# Get database URL from environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")

# Create engine
engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

# Create base class for models
Base = declarative_base()

class User(Base):
    """User model for authentication."""
    
    __tablename__ = "users"
    
    id = Column(String(32), primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    transactions = relationship("Transaction", back_populates="user", cascade="all, delete-orphan")
    goals = relationship("Goal", back_populates="user", cascade="all, delete-orphan")
    budgets = relationship("Budget", back_populates="user", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

class Transaction(Base):
    """Transaction model for financial transactions."""
    
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(32), ForeignKey("users.id"), nullable=False)
    date = Column(String(10), nullable=False)  # YYYY-MM-DD format
    description = Column(String(255), nullable=False)
    amount = Column(Float, nullable=False)
    type = Column(String(10), nullable=False)  # income or expense
    category = Column(String(50), nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="transactions")
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "date": self.date,
            "description": self.description,
            "amount": self.amount,
            "type": self.type,
            "category": self.category,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

class Goal(Base):
    """Goal model for financial goals."""
    
    __tablename__ = "goals"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(32), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    target_amount = Column(Float, nullable=False)
    current_amount = Column(Float, default=0.0)
    deadline = Column(String(10), nullable=False)  # YYYY-MM-DD format
    category = Column(String(50), nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="goals")
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "target_amount": self.target_amount,
            "current_amount": self.current_amount,
            "deadline": self.deadline,
            "category": self.category,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

class Budget(Base):
    """Budget model for financial budgets."""
    
    __tablename__ = "budgets"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(32), ForeignKey("users.id"), nullable=False)
    category = Column(String(50), nullable=False)
    amount = Column(Float, nullable=False)
    period = Column(String(10), nullable=False)  # monthly, weekly, yearly
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="budgets")
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "category": self.category,
            "amount": self.amount,
            "period": self.period,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

class VectorTransaction(Base):
    """VectorTransaction model for vector-based transaction analysis."""
    
    __tablename__ = "vector_transactions"
    
    id = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)
    embedding = Column(JSON, nullable=True)  # Stored as JSON array
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class FinancialAnalysis(Base):
    """FinancialAnalysis model for storing AI-generated analyses."""
    
    __tablename__ = "financial_analyses"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(32), ForeignKey("users.id"), nullable=False)
    analysis_type = Column(String(50), nullable=False)  # insight, trend, prediction, etc.
    content = Column(Text, nullable=False)
    meta_data = Column(JSON, nullable=True)  # Additional data as JSON (renamed from metadata to avoid SQLAlchemy reserved keyword)
    created_at = Column(DateTime, default=func.now())

class UserSetting(Base):
    """UserSetting model for user preferences and settings."""
    
    __tablename__ = "user_settings"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(32), ForeignKey("users.id"), nullable=False, unique=True)
    notification_enabled = Column(Boolean, default=True)
    theme = Column(String(20), default="dark")
    currency = Column(String(3), default="USD")
    settings_data = Column(JSON, nullable=True)  # Flexible settings as JSON
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# Create all tables
def init_db():
    """Initialize the database."""
    Base.metadata.create_all(bind=engine)

# Get a database session
def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

if __name__ == "__main__":
    init_db()