from sqlalchemy.orm import sessionmaker, declarative_base, Session
import os
from sqlalchemy import create_engine, Column, Integer, String

database_url = os.getenv("database_url")

engine = create_engine(database_url,pool_pre_ping=True)

sessionlocal = sessionmaker(autocommit=False, 
autoflush=False, bind=engine)
Base = declarative_base()