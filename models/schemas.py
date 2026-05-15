from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class ContextRequest(BaseModel):
    user_id: str
    message: str
    conversation_history: list[dict] = []


class ContextResponse(BaseModel):
    system_injection: str
    recommended_model: str   # "local" or "openai"
    lora_id: Optional[str] = None
    confidence: float
    loop_state: dict = Field(default_factory=dict)


class SaveRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    user_id: str
    user_message: str
    assistant_message: str
    model_used: str
    engagement_signal: str = "continue"  # thumbs_up | thumbs_down | continue | quiet
