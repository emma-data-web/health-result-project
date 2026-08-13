from pydantic import BaseModel, ConfigDict, Field

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