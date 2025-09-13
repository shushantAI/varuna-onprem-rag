# database.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Point at a local file called "users.db" in your project folder.
DATABASE_URL = "sqlite+aiosqlite:///./users.db"

# Create an async engine that will automatically create/open users.db
engine = create_async_engine(DATABASE_URL, echo=False, future=True)

# Every call to AsyncSessionLocal() yields a new AsyncSession that talks to SQLite.
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
