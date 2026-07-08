import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_runtime import ADAPTER_ID, MODEL_ID, get_runtime


class Message(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[Message] = Field(default_factory=list, max_length=12)


class ChatResponse(BaseModel):
    answer: str
    model: str


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("API_KEY")
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("LOAD_MODEL_ON_STARTUP", "true").lower() == "true":
        get_runtime()
    yield


app = FastAPI(
    title="CourseLM model API",
    version="2.0.0",
    lifespan=lifespan,
)
origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID, "adapter": ADAPTER_ID}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(request: ChatRequest) -> ChatResponse:
    history = [message.model_dump() for message in request.history]
    answer = get_runtime().generate(request.message, history)
    return ChatResponse(answer=answer, model=MODEL_ID)
