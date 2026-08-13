from pydantic import BaseModel, ConfigDict, Field



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