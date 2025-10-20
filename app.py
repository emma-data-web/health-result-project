from fastapi import FastAPI, status, HTTPException, Depends
from pydantic import BaseModel, ConfigDict
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base, Session


app = FastAPI()


database_url = "sqlite:///./test.db"
engine = create_engine(database_url, 
connect_args={"check_same_thread":False})
sessionlocal = sessionmaker(autocommit=False, 
autoflush=False, bind=engine)
Base = declarative_base()


class UserDb(Base):
    __tablename__ = "users"

    id = Column(Integer,primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)

#Base.metadata.create_all(bind=engine)   #---- create table


class UserCreate(BaseModel): # --request model
    name: str
    email : str

class UserResponse(BaseModel): # -- response model
    id: int
    name: str
    email: str
    
    model_config = ConfigDict(from_attributes=True)


def get_db():
    db = sessionlocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    print("Tables created.")



@app.post("/users/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if email already exists
    existing_user = db.query(UserDb).filter(UserDb.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_user = UserDb(name=user.name, email=user.email)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)  # Refresh instance to get the new id
    
    return new_user