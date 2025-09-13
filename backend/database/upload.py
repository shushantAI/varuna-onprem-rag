import asyncio
from .database import engine, AsyncSessionLocal
from .models import Base
from .crud import create_user, get_user_by_username

async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as db:
        with open("test_users.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    print(f"Skipping invalid line: {line!r}")
                    continue

                username, password = parts[0], parts[1]

                #Check if this username already exists
                existing_user = await get_user_by_username(db, username)
                if existing_user:
                    print(f"User '{username}' already exists, skipping.")
                    continue

                # Create and commit the new user
                new_user = await create_user(db, username, password)
                print(f"Created user: {new_user.username} (id={new_user.id})")

if __name__ == "__main__":
    asyncio.run(main())
