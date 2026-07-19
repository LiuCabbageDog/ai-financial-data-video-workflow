"""Unit and integration checks for the highest-risk financial workflow rules."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
from app.main import app
from app.models import JobInput
from app.pipeline import Pipeline, ROOT, normalize_spoken
from app.config import Settings


def sample() -> JobInput:
    """Load the versioned NVIDIA sample request."""
    return JobInput.model_validate_json((ROOT/"examples/nvidia_q1_fy27/input.json").read_text())


def test_spoken_text_normalizes_financial_notation() -> None:
    """Ensure display notation is never passed verbatim to Mandarin TTS."""
    spoken=normalize_spoken("Q1 FY2027 收入 $81.6B，同比 +85%")
    assert "$" not in spoken and "FY" not in spoken and "百分之85" in spoken


def test_pre_written_conflict_blocks() -> None:
    """An unsupported numeric claim must create a conflict report."""
    req=sample(); req.transcript.mode="pre-written"; req.transcript.text="收入同比增长 999%"
    pipeline=Pipeline("test-conflict",req,auto_approve=True)
    facts=json.loads((ROOT/"examples/nvidia_q1_fy27/canonical_facts.json").read_text())
    assert pipeline._conflict_check(facts)["unsupported_percentages"] == [999.0]


def test_omitted_transcript_defaults_to_generation() -> None:
    """Clients may omit transcript when they want the workflow to generate it."""
    payload=sample().model_dump(mode="json"); payload.pop("transcript")
    request=JobInput.model_validate(payload)
    assert request.transcript.mode == "generate" and request.transcript.text is None


def test_locked_pre_written_transcript_is_used_as_narration() -> None:
    """A non-editable supplied script must reach narration instead of being replaced."""
    req=sample(); req.transcript.mode="pre-written"; req.transcript.text="  第一句。\n第二句。  "; req.transcript.allow_editing=False
    pipeline=Pipeline("test-pre-written",req,auto_approve=True)
    narration=pipeline._resolve_narration(pipeline._scenes(),{"narration":{"segments":[]}})
    content="".join(segment["display_text"] for segment in narration["segments"][:-1])
    assert content == req.transcript.text
    assert narration["source"] == "pre-written" and narration["editing_applied"] is False


def test_api_idempotency_and_scene_retry(tmp_path: Path) -> None:
    """Repeated clicks deduplicate and retries preserve upstream artifacts."""
    client=TestClient(app); payload=sample().model_dump(mode="json"); headers={"Idempotency-Key":"pytest-fixed-key"}
    first=client.post("/api/jobs",json=payload,headers=headers); second=client.post("/api/jobs",json=payload,headers=headers)
    assert first.status_code==202 and first.json()["job_id"]==second.json()["job_id"] and second.json()["deduplicated"]


def test_canonical_facts_have_time_unit_and_source() -> None:
    """Every canonical fact carries ambiguity-resolving metadata."""
    data=json.loads((ROOT/"examples/nvidia_q1_fy27/canonical_facts.json").read_text())
    for fact in data["facts"]:
        assert fact["unit"] and fact["scale"] and fact["fiscal_period"] and fact["source_locator"]


def test_deterministic_mode_requires_no_provider_keys(monkeypatch) -> None:
    """The zero-cost mode must remain runnable without external credentials."""
    for key in ("OPENAI_API_KEY","ELEVENLABS_API_KEY","ELEVENLABS_VOICE_ID","BACH_API_KEY","BACH_BASE_URL"):
        monkeypatch.delenv(key,raising=False)
    settings=Settings.load("deterministic")
    settings.validate()
    assert settings.mode == "deterministic"


def test_production_mode_fails_before_api_calls_when_keys_are_missing(monkeypatch) -> None:
    """Production must never silently fall back to deterministic providers."""
    for key in ("OPENAI_API_KEY","ELEVENLABS_API_KEY","ELEVENLABS_VOICE_ID","BACH_API_KEY","BACH_BASE_URL"):
        monkeypatch.delenv(key,raising=False)
    settings=Settings.load("production")
    try:
        settings.validate()
    except ValueError as error:
        message=str(error)
        assert "OPENAI_API_KEY" in message and "BACH_BASE_URL" in message
    else:
        raise AssertionError("production configuration unexpectedly validated")
