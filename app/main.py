import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.agent import SHLAgent
from app.models import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Lifespan: initialise the agent (loads model + FAISS) once at startup
# --------------------------------------------------------------------------- #
agent: SHLAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("Initialising SHL agent…")
    agent = SHLAgent()
    logger.info("SHL agent ready.")
    yield
    logger.info("Shutting down.")


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent that recommends SHL assessments based on role "
        "requirements through multi-turn dialogue."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, summary="Readiness check")
def health():
    """Returns 200 OK when the service is ready."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, summary="Chat with the recommender")
async def chat(request: ChatRequest):
    """
    Stateless endpoint: send the full conversation history, receive the next agent reply.

    - `messages`: list of `{role, content}` pairs (alternating user / assistant).
    - `reply`: agent's next message.
    - `recommendations`: null while clarifying; list of 1–10 items when committed.
    - `end_of_conversation`: true when the agent considers the task complete.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    # Enforce turn cap (evaluator limit: 8 turns = 4 user + 4 assistant)
    if len(request.messages) > 16:
        raise HTTPException(status_code=400, detail="Conversation exceeds maximum turn limit")

    try:
        response = await agent.chat(request.messages)
        return response
    except Exception as e:
        logger.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error")
