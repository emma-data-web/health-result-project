from sqlalchemy import  Column, Integer, String
from app.db.config import Base


class UserDb(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    position = Column(String, nullable=True)
    department = Column(String, nullable=True)