from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    try:
        print("Password before hashing:", repr(password))
        print("Password byte length:", len(password.encode('utf-8')))
        return pwd_context.hash(password)
    except Exception as e:
        print("Error in get_password_hash:", e)
        raise
