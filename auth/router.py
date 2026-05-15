import uuid
import logging
import bcrypt
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from db.database import get_conn
from auth.jwt_utils import create_token

router = APIRouter(prefix="/auth", tags=["auth"])
log = logging.getLogger("echo.auth")


def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str
    existing_user_id: str | None = None  # pass to claim an existing data UUID


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user_id: str
    username: str


@router.post("/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    async with get_conn() as db:
        async with db.execute(
            "SELECT id FROM users WHERE email = ? OR username = ?",
            (req.email.lower(), req.username)
        ) as cur:
            if await cur.fetchone():
                raise HTTPException(400, "Email or username already taken")
        user_id = req.existing_user_id or str(uuid.uuid4())
        hashed = _hash(req.password)
        await db.execute(
            "INSERT INTO users (id, email, username, password) VALUES (?, ?, ?, ?)",
            (user_id, req.email.lower(), req.username, hashed),
        )
        await db.commit()
    log.info("Registered user=%s email=%s", user_id, req.email)
    return AuthResponse(token=create_token(user_id), user_id=user_id, username=req.username)


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    async with get_conn() as db:
        async with db.execute(
            "SELECT id, username, password FROM users WHERE email = ?",
            (req.email.lower(),)
        ) as cur:
            row = await cur.fetchone()
    if not row or not _verify(req.password, row["password"]):
        raise HTTPException(401, "Invalid email or password")
    log.info("Login user=%s", row["id"])
    return AuthResponse(token=create_token(row["id"]), user_id=row["id"], username=row["username"])


@router.get("/me")
async def me(request: Request):
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(401, "Unauthorized")
    async with get_conn() as db:
        async with db.execute(
            "SELECT id, email, username, created_at FROM users WHERE id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    if not row:
        raise HTTPException(404, "User not found")
    return dict(row)
