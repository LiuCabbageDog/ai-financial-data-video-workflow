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
    elevenlabs_speed: float
    mandarin_chars_per_second: float
    narration_planning_tolerance: float
    target_duration_tolerance: float
    narration_rewrite_attempts: int
    tts_auto_fit_duration: bool
    elevenlabs_stability: float
    elevenlabs_similarity_boost: float
    elevenlabs_style: float
    elevenlabs_use_speaker_boost: bool
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
            elevenlabs_speed=float(os.getenv("ELEVENLABS_SPEED","1.02")),
            mandarin_chars_per_second=float(os.getenv("MANDARIN_CHARS_PER_SECOND","4.4")),
            narration_planning_tolerance=float(os.getenv("NARRATION_PLANNING_TOLERANCE","0.10")),
            target_duration_tolerance=float(os.getenv("TARGET_DURATION_TOLERANCE","0.15")),
            narration_rewrite_attempts=int(os.getenv("NARRATION_REWRITE_ATTEMPTS","3")),
            tts_auto_fit_duration=os.getenv("TTS_AUTO_FIT_DURATION","true").strip().lower() in {"1","true","yes","on"},
            elevenlabs_stability=float(os.getenv("ELEVENLABS_STABILITY","0.45")),
            elevenlabs_similarity_boost=float(os.getenv("ELEVENLABS_SIMILARITY_BOOST","0.75")),
            elevenlabs_style=float(os.getenv("ELEVENLABS_STYLE","0")),
            elevenlabs_use_speaker_boost=os.getenv("ELEVENLABS_USE_SPEAKER_BOOST","true").strip().lower() in {"1","true","yes","on"},
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

    def validate(self, voice_id_override: str | None = None) -> None:
        """Fail before billable work if production credentials are incomplete."""
        if self.mode == "deterministic": return
        missing=[]
        for name,value in [("OPENAI_API_KEY",self.openai_api_key),("ELEVENLABS_API_KEY",self.elevenlabs_api_key),("ELEVENLABS_VOICE_ID",voice_id_override or self.elevenlabs_voice_id),("BACH_ACCESS_KEY",self.bach_access_key),("BACH_SECRET_KEY",self.bach_secret_key),("BACH_BASE_URL",self.bach_base_url)]:
            if not value: missing.append(name)
        if missing: raise ValueError("Production mode is missing required configuration: "+", ".join(missing))
        if not self.elevenlabs_model_id: raise ValueError("ELEVENLABS_MODEL_ID must not be empty")
        if not self.elevenlabs_output_format: raise ValueError("ELEVENLABS_OUTPUT_FORMAT must not be empty")
        if not .7 <= self.elevenlabs_speed <= 1.2: raise ValueError("ELEVENLABS_SPEED must be between 0.7 and 1.2")
        if self.mandarin_chars_per_second <= 0: raise ValueError("MANDARIN_CHARS_PER_SECOND must be positive")
        if not 0 < self.narration_planning_tolerance <= .25: raise ValueError("NARRATION_PLANNING_TOLERANCE must be between 0 and 0.25")
        if not 0 < self.target_duration_tolerance <= .25: raise ValueError("TARGET_DURATION_TOLERANCE must be between 0 and 0.25")
        if not 1 <= self.narration_rewrite_attempts <= 5: raise ValueError("NARRATION_REWRITE_ATTEMPTS must be between 1 and 5")
        for name,value in (("ELEVENLABS_STABILITY",self.elevenlabs_stability),("ELEVENLABS_SIMILARITY_BOOST",self.elevenlabs_similarity_boost),("ELEVENLABS_STYLE",self.elevenlabs_style)):
            if not 0 <= value <= 1: raise ValueError(f"{name} must be between 0 and 1")
        if self.bach_resolution not in {"720p","1080p"}: raise ValueError("BACH_RESOLUTION must be 720p or 1080p")
        if self.bach_poll_interval_seconds <= 0: raise ValueError("BACH_POLL_INTERVAL_SECONDS must be positive")
