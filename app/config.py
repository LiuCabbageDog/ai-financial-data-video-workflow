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
    elevenlabs_model_id: str
    elevenlabs_output_format: str
    bach_access_key: str | None
    bach_secret_key: str | None
    bach_base_url: str
    bach_text_to_video_path: str
    bach_elements_to_video_path: str
    bach_text_to_image_path: str
    bach_text_to_subject_path: str
    bach_model_name: str
    bach_resolution: str
    bach_poll_interval_seconds: float
    request_timeout_seconds: float
    openai_input_usd_per_million: float
    openai_output_usd_per_million: float
    elevenlabs_usd_per_thousand_chars: float
    bach_video_usd_per_call: float
    bach_image_usd_per_call: float

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
            elevenlabs_model_id=os.getenv("ELEVENLABS_MODEL_ID","eleven_multilingual_v2").strip(),
            elevenlabs_output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT","mp3_44100_128").strip(),
            bach_access_key=os.getenv("BACH_ACCESS_KEY") or None,
            bach_secret_key=os.getenv("BACH_SECRET_KEY") or None,
            bach_base_url=os.getenv("BACH_BASE_URL","https://api-gen-na.bach.art/api/vdr").rstrip("/"),
            bach_text_to_video_path=os.getenv("BACH_TEXT_TO_VIDEO_PATH","/videos/text2video"),
            bach_elements_to_video_path=os.getenv("BACH_ELEMENTS_TO_VIDEO_PATH","/videos/elements2video"),
            bach_text_to_image_path=os.getenv("BACH_TEXT_TO_IMAGE_PATH","/images/text2image"),
            bach_text_to_subject_path=os.getenv("BACH_TEXT_TO_SUBJECT_PATH","/subject/text2image"),
            bach_model_name=os.getenv("BACH_MODEL_NAME","bach-1.0-preview"),
            bach_resolution=os.getenv("BACH_RESOLUTION","1080p"),
            bach_poll_interval_seconds=float(os.getenv("BACH_POLL_INTERVAL_SECONDS","10")),
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS","120")),
            openai_input_usd_per_million=float(os.getenv("OPENAI_INPUT_USD_PER_MILLION","0")),
            openai_output_usd_per_million=float(os.getenv("OPENAI_OUTPUT_USD_PER_MILLION","0")),
            elevenlabs_usd_per_thousand_chars=float(os.getenv("ELEVENLABS_USD_PER_THOUSAND_CHARS","0")),
            bach_video_usd_per_call=float(os.getenv("BACH_VIDEO_USD_PER_CALL","0")),
            bach_image_usd_per_call=float(os.getenv("BACH_IMAGE_USD_PER_CALL","0")),
        )

    def validate(self) -> None:
        """Fail before billable work if production credentials are incomplete."""
        if self.mode == "deterministic": return
        missing=[]
        for name,value in [("OPENAI_API_KEY",self.openai_api_key),("ELEVENLABS_API_KEY",self.elevenlabs_api_key),("ELEVENLABS_VOICE_ID",self.elevenlabs_voice_id),("BACH_ACCESS_KEY",self.bach_access_key),("BACH_SECRET_KEY",self.bach_secret_key),("BACH_BASE_URL",self.bach_base_url)]:
            if not value: missing.append(name)
        if missing: raise ValueError("Production mode is missing required configuration: "+", ".join(missing))
        if not self.elevenlabs_model_id: raise ValueError("ELEVENLABS_MODEL_ID must not be empty")
        if not self.elevenlabs_output_format: raise ValueError("ELEVENLABS_OUTPUT_FORMAT must not be empty")
        if self.bach_resolution not in {"720p","1080p"}: raise ValueError("BACH_RESOLUTION must be 720p or 1080p")
        if self.bach_poll_interval_seconds <= 0: raise ValueError("BACH_POLL_INTERVAL_SECONDS must be positive")
