from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from config import settings

ALGORITHM = "HS256"


def create_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.token_expire_days)
    return jwt.encode({"sub": user_id, "exp": expire}, settings.jwt_secret, algorithm=ALGORITHM)


def verify_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        return user_id if user_id else None
    except JWTError:
        return None
