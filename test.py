from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta
from fastapi import Depends, Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI()

# Secret key to encode JWT
SECRET_KEY = "emma690"

# Mock user (replace with your DB in real app)
mock_user = {
    "username": "user1",
    "password": "password123",  # Don't store plaintext passwords in production!
    "id": 101
}

# Request body schema for login
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/logins")
def login(data: LoginRequest):
    if data.username == mock_user["username"] and data.password == mock_user["password"]:
        # Create JWT payload with expiration
        payload = {
            "user_id": mock_user["id"],
            "username": mock_user["username"],
            "exp": datetime.utcnow() + timedelta(hours=1)  # Token expires in 1 hour
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        return {"token": token}

    raise HTTPException(status_code=401, detail="Invalid username or password")



security = HTTPBearer()

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials  # Extract token string from "Bearer <token>"
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload  # You can return user info from here
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")



@app.get("/profile")
def get_profile(user_data: dict = Depends(verify_jwt)):
    return {
        "message": "Welcome to your profile!",
        "user_id": user_data["user_id"],
        "username": user_data["username"]
    }
