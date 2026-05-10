from pydantic import BaseModel, Field
from typing import List, Optional


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str
    recommendations: Optional[List[Recommendation]] = None
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str
