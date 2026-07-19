"""Environment-backed runtime configuration for deterministic and production modes."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Resolved workflow settings; production credentials are validated eagerly."""
    mode: Literal["deterministic", "production"]
    artifact_root: Path
    openai_api_key: str | None
    openai_model: str
    openai_base_url: str
    elevenlabs_api_key: str | None
    elevenlabs_voice_id: str | None
    elevenlabs_base_url: str
    bach_api_key: str | None
    bach_base_url: str | None
    bach_video_path: str
    bach_image_path: str
    request_timeout_seconds: float

    @classmethod
    def load(cls, mode_override: str | None = None) -> "Settings":
        """Load `.env`, apply an optional CLI override, and reject invalid modes."""
        load_dotenv()
        mode=(mode_override or os.getenv("ADAPTER_MODE","deterministic")).strip().lower()
        if mode not in {"deterministic","production"}:
            raise ValueError("ADAPTER_MODE must be 'deterministic' or 'production'")
        return cls(
            mode=mode, artifact_root=Path(os.getenv("ARTIFACT_ROOT","./runs")).resolve(),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL","gpt-5-mini"),
            openai_base_url=os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1").rstrip("/"),
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY") or None,
            elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID") or None,
            elevenlabs_base_url=os.getenv("ELEVENLABS_BASE_URL","https://api.elevenlabs.io/v1").rstrip("/"),
            bach_api_key=os.getenv("BACH_API_KEY") or None,
            bach_base_url=(os.getenv("BACH_BASE_URL") or "").rstrip("/") or None,
            bach_video_path=os.getenv("BACH_VIDEO_PATH","/v1/generate/video"),
            bach_image_path=os.getenv("BACH_IMAGE_PATH","/v1/generate/image"),
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS","120")),
        )

    def validate(self) -> None:
        """Fail before billable work if production credentials are incomplete."""
        if self.mode == "deterministic": return
        missing=[]
        for name,value in [("OPENAI_API_KEY",self.openai_api_key),("ELEVENLABS_API_KEY",self.elevenlabs_api_key),("ELEVENLABS_VOICE_ID",self.elevenlabs_voice_id),("BACH_API_KEY",self.bach_api_key),("BACH_BASE_URL",self.bach_base_url)]:
            if not value: missing.append(name)
        if missing: raise ValueError("Production mode is missing required configuration: "+", ".join(missing))
