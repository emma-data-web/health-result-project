from fastapi import FastAPI, status, HTTPException, Depends
from typing import Optional
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



model = joblib.load("real_malaria_model.pkl")

dia_model = joblib.load("diabetes_model.pkl")

health_model = joblib.load("health_disease_pipeline.joblib")
health_label_encoder = joblib.load("label_encoder.joblib")




@app.on_event("startup")
def on_startup():
    print(" Dropping and recreating tables...")
   #Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print(" Tables created successfully")


#this is a very basic auth system, use tokens if you want to use the project properly.
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
        traceback.print_exc()  
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}  
        )
    

@app.post("/login")
def user_login(user: UserLogin, db: Session = Depends(get_db)):
        
          
        db_user = db.query(UserDb).filter(UserDb.email == user.email).first()
        if not db_user:
            raise HTTPException(status_code=400, detail="Invalid email or password")
        
        
        if not pwd_context.verify(user.password, db_user.hashed_password):
            raise HTTPException(status_code=400, detail="Invalid email or password")
        
        
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

         input_data = pd.DataFrame([features.model_dump(by_alias=True)])

       
         prediction = model.predict(input_data)

        
         return {"death": int(prediction[0])}

    except Exception as e:
    
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/malpredict", response_model=ModelResponse, status_code=status.HTTP_200_OK)
def get_malaria_prediction(user: ModelRequest):
    try:
        
        input_data = pd.DataFrame([user.model_dump(by_alias=True)])

        
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


