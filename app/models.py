"""Typed domain models shared by the API and deterministic pipeline."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class JobStatus(str, Enum):
    """Explicit externally visible asynchronous job states."""
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_REVIEW = "waiting_review"
    RENDERING = "rendering"
    QA = "qa"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED_CONFLICT = "blocked_conflict"


class Transcript(BaseModel):
    """Transcript mode and language with cross-field validation."""
    mode: Literal["generate", "pre-written"] = "generate"
    text: str | None = None
    allow_editing: bool = True
    language: str = "zh-CN"

    @model_validator(mode="after")
    def require_pre_written_text(self) -> "Transcript":
        """Reject a pre-written mode without actual text."""
        if self.mode == "pre-written" and not (self.text or "").strip():
            raise ValueError("pre-written mode requires transcript.text")
        return self


class JobInput(BaseModel):
    """Normalized input envelope; omitted groups receive safe defaults."""
    source_materials: dict[str, Any]
    content_requirements: dict[str, Any] | None = None
    transcript: Transcript = Field(default_factory=Transcript)
    creative_direction: dict[str, Any] = Field(default_factory=lambda: {"audience":"retail investors","visual_style":"cartoon"})
    audio: dict[str, Any] = Field(default_factory=lambda: {"provider":"elevenlabs","language":"zh-CN"})
    captions: dict[str, Any] = Field(default_factory=lambda: {"enabled":True,"max_lines":2,"position":"bottom"})
    brand: dict[str, Any] = Field(default_factory=lambda: {"primary_color":"#76B900","font_family":"Inter"})
    output: dict[str, Any] = Field(default_factory=lambda: {"resolution":"1920x1080","fps":30,"format":"mp4"})


class JobRecord(BaseModel):
    """Persisted job state returned to clients."""
    id: str
    status: JobStatus
    current_node: str | None = None
    progress: float = 0
    retries: int = 0
    error: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    nodes: dict[str, str] = Field(default_factory=dict)


class ReviewRequest(BaseModel):
    """Reviewer decision and optional note."""
    note: str = ""


class RetryRequest(BaseModel):
    """Scene retry override; empty means reuse the prior request."""
    reason: str = "manual retry"
    overrides: dict[str, Any] = Field(default_factory=dict)
