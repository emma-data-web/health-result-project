from fastapi import FastAPI, status, HTTPException, Depends
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from passlib.context import CryptContext
import os
import traceback
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

#print("Using DB file:", os.path.abspath("./test.db"))

database_url = os.getenv("database_url")

engine = create_engine(database_url)

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

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    position = Column(String, nullable=True)
    department = Column(String, nullable=True)
#Base.metadata.create_all(bind=engine)   #---- create table


class UserCreate(BaseModel): # --request model
    name: str
    email : str
    password: str
    position: str
    department: str

class UserResponse(BaseModel): # -- response model
    id: int
    name: str
    email: str
    
    model_config = ConfigDict(from_attributes=True)

class UserLogin(BaseModel):
    email: str
    password: str

class UserProfileRequest(BaseModel):
    name: str
    email: str

class UserProfileResponse(BaseModel):
    name: str
    email: str
    position: str
    Department: str

class UserFeaturesRequest(BaseModel):
    country: str = Field(alias="Country")
    year: int = Field(alias="Year")
    no_of_cases_median: int = Field(alias="No. of cases_median")
    no_of_cases_min: int = Field(alias="No. of cases_min")
    no_of_cases_max: int = Field(alias="No. of cases_max")
    no_of_deaths_median: int = Field(alias="No. of deaths_median")
    no_of_deaths_min: int = Field(alias="No. of deaths_min")
    no_of_deaths_max: int = Field(alias="No. of deaths_max")
    who_region: str = Field(alias="WHO Region")
    case: int = Field(alias="case")

class UserTragetResponse(BaseModel):
    death: int

def get_db():
    db = sessionlocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    print("🔁 Dropping and recreating tables...")
   #Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully")



@app.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED, include_in_schema=True)
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
        hashed_password=hashed_password,
        position=user.position,
        department=user.department
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
    

@app.post("/login")
def user_login(user: UserLogin, db: Session = Depends(get_db)):
        
          # 1. Find user by email
        db_user = db.query(UserDb).filter(UserDb.email == user.email).first()
        if not db_user:
            raise HTTPException(status_code=400, detail="Invalid email or password")
        
        # 2. Verify password
        if not pwd_context.verify(user.password, db_user.hashed_password):
            raise HTTPException(status_code=400, detail="Invalid email or password")
        
        # 3. Return success message
        return {"message": f"Welcome back, {db_user.name}!"}
    

@app.post("/profile", response_model=UserProfileResponse, status_code=status.HTTP_200_OK)
def check_profile(user: UserProfileRequest, db: Session = Depends(get_db)):
    db_user = db.query(UserDb).filter(UserDb.email == user.email).first()
    if not db_user:
        raise HTTPException(status_code=400, detail="pls login to acess your profile")
    
    profile = {
        "name": db_user.name,
        "email": db_user.email,
        "position": db_user.position,
        "Department": db_user.department
    }

    return profile


@app.post("/predict", response_model=UserTragetResponse,status_code=status.HTTP_200_OK)
def get_predictions(features: UserFeaturesRequest):
    try:

         model = joblib.load("malaria_death_mmodel.pkl")

        # Convert input to DataFrame using the Pydantic model's dict (with aliases)
         input_data = pd.DataFrame([features.model_dump(by_alias=True)])

        # Make prediction
         prediction = model.predict(input_data)

        # Return the prediction
         return {"death": int(prediction[0])}

    except Exception as e:
    
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

