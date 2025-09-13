# crud.py
from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from passlib.context import CryptContext

from database.models import User

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(plain_password: str) -> str:
    return pwd_ctx.hash(plain_password)

def verify_password(plain_password: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain_password, hashed)

async def create_user(db, username: str, password: str):
    hashed = get_password_hash(password)
    new_user = User(username=username, password_hash=hashed)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user

async def get_user_by_username(db, username: str):
    result = await db.execute(select(User).where(User.username == username))
    return result.scalars().first()

async def authenticate_user(db, username: str, password: str):
    user = await get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user
