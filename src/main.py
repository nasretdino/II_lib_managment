from contextlib import asynccontextmanager
from fastapi import FastAPI


from src.db import engine
from src.modules import user_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)
app.include_router(user_router)

