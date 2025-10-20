from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


app = FastAPI()


database_url = "sqlite://./test.db"
engine = create_engine(database_url, 
connect_args={"check_same_thread":False})
sessionlocal = sessionmaker(autocommit=False, 
autoflush=False, bind=engine)
Base = declarative_base()


class userdb(Base):
    __tablename__ = "users"







class User(BaseModel):   # request model
    name: str
    age: int
    email: str


class UserResponse(BaseModel):  # response model
    id:int
    name: str
    email: str


@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(user: User):
    fake_id = 1
    return {
        "id":fake_id,
        "name": user.name,
        "email": user.email
    }


@app.post("/user")
def create(user: User):
    return{
        "message": "user created",
        "user_data": user
    }

@app.post("/new")
def fast(name="emma"):
    return {
        "msg": name
    }


@app.get("/test")
def test():
    return {"its working fine"}


@app.get("/try/{user_id}")
def new(user_id: int):
    return{"the result": {user_id}}


@app.get("/search")
def away(name: str, age: int):
    return {"the name": name, "the age": age}