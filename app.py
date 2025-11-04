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
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()


app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"],
)
database_url = os.getenv("database_url")

engine = create_engine(database_url,pool_pre_ping=True)

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

class ModelRequest(BaseModel):
    Age: int
    Body_Temperature: float
    Hemoglobin: float
    RBC_Count: float
    Platelet_Count: int
    Has_Fever: int
    Has_Chills: int
    Has_Vomiting: int
    Rainy_Season: int

class ModelResponse(BaseModel):
    Result: int
    confidence: str




class DiaModelRequest(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


class DiaModelResponse(BaseModel):
    Outcome: int


class HealthModelRequest(BaseModel):
    age: int
    gender: str
    temperature: float
    heart_rate: int
    systolic_bp: int
    diastolic_bp: int
    glucose_level: float
    oxygen_level: float
    bmi: float
    cough: str
    fatigue: str
    headache: str
    nausea: str
    chest_pain: str
    shortness_of_breath: str
    vision_problem: str
    frequent_urination: str
    joint_pain: str


class HealthModelResponse(BaseModel):
    predicted_disease: str


model = joblib.load("real_malaria_model.pkl")

dia_model = joblib.load("diabetes_model.pkl")

health_model = joblib.load("health_disease_pipeline.joblib")
health_label_encoder = joblib.load("label_encoder.joblib")

def get_db():
    db = sessionlocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    print(" Dropping and recreating tables...")
   #Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print(" Tables created successfully")



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


@app.post("/malpredict", response_model=ModelResponse, status_code=status.HTTP_200_OK)
def get_malaria_prediction(user: ModelRequest):
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([user.model_dump(by_alias=True)])

        # Make prediction (0 or 1)
        prediction = model.predict(input_data)

        
        confidence = round(model.predict_proba(input_data)[:, 1][0] * 100)

        

        return {
            "Result": int(prediction[0]),
            "confidence":  f" the confidences level of the result is {confidence}%"
        }

    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")



@app.post("/diapredict", response_model=DiaModelResponse, status_code=status.HTTP_200_OK)
def get_dia(user: DiaModelRequest):
    try: 
        input = pd.DataFrame([user.model_dump(by_alias=True)])

        prediction = dia_model.predict(input)

        return {"Outcome": int(prediction)}
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    


@app.post("/healthpredict", response_model=HealthModelResponse, status_code=status.HTTP_200_OK)
def predict_health_condition(data: HealthModelRequest):
    try:
        
        input_df = pd.DataFrame([data.model_dump()])

        
        pred_encoded = health_model.predict(input_df)
        pred_label = health_label_encoder.inverse_transform(pred_encoded)[0]

        return {"predicted_disease": pred_label}

    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

print(" Using database:", database_url)
