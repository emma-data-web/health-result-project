from fastapi import FastAPI, status, HTTPException, Depends
from pydantic import BaseModel, ConfigDict
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from passlib.context import CryptContext
import os
import traceback
from fastapi.responses import JSONResponse


app = FastAPI()

print("Using DB file:", os.path.abspath("./test.db"))
database_url = "sqlite:///./test.db"
engine = create_engine(database_url, 
connect_args={"check_same_thread":False})
sessionlocal = sessionmaker(autocommit=False, 
autoflush=False, bind=engine)
Base = declarative_base()


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    try:
        print("Password before hashing:", repr(password))
        print("Password byte length:", len(password.encode('utf-8')))
        return pwd_context.hash(password)
    except Exception as e:
        print("Error in get_password_hash:", e)
        raise

class UserDb(Base):
    __tablename__ = "users"

    id = Column(Integer,primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

#Base.metadata.create_all(bind=engine)   #---- create table


class UserCreate(BaseModel): # --request model
    name: str
    email : str
    password: str

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
    try:
        # Check if user already exists
        existing_user = db.query(UserDb).filter(UserDb.email == user.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = get_password_hash(user.password)

        new_user = UserDb(
            name=user.name,
            email=user.email,
            hashed_password=hashed_password
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user

    except Exception as e:
        traceback.print_exc()  # Print error to console
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}  # Send error message in response
        )