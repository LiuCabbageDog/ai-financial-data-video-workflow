"""Unit, integration, and contract tests for high-risk workflow behavior."""
from __future__ import annotations

import base64, json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.adapters import BachAssets, ElevenLabsTTS
from app.main import app
from app.models import CanonicalFacts, JobInput, JobStatus, PlanningBundle
from app.pipeline import Pipeline, ROOT, normalize_spoken


def sample() -> JobInput:
    """Load the versioned NVIDIA request."""

    return JobInput.model_validate_json((ROOT/"examples/nvidia_q1_fy27/input.json").read_text())


def local_settings(tmp_path:Path) -> Settings:
    """Return isolated deterministic settings for one test."""

    return replace(Settings.load("deterministic"),artifact_root=tmp_path)


def fake_render(self:Pipeline) -> None:
    """Create a non-empty MP4 stand-in without invoking Chromium."""

    (self.run_dir/"final.mp4").write_bytes(b"video"*512)
    if "final.mp4" not in self.record.artifacts: self.record.artifacts.append("final.mp4")


def fake_tts(self:Pipeline,narration:dict) -> dict:
    """Return a no-audio deterministic TTS artifact for state-machine tests."""

    public=ROOT/"public"; public.mkdir(exist_ok=True); name=f"{self.job_id}-test.wav"; (public/name).write_bytes(b"audio")
    return {"provider":"test","characters":sum(len(x["spoken_text"]) for x in narration["segments"]),"artifact":name,"remotion_static_src":name}


def approve(run:Path,node:str) -> None:
    """Persist an approval exactly as the HTTP review endpoint would."""

    (run/f"review_{node}.json").write_text(json.dumps({"node":node,"status":"approved","reviewer":"pytest"}))


def test_spoken_text_normalizes_generic_financial_notation() -> None:
    """Normalization must work beyond hard-coded NVIDIA values."""

    spoken=normalize_spoken("Q3 FY2028 non-GAAP 收入 $54.3B，同比增长 +12%，提升 50 bps")
    assert "$" not in spoken and "FY" not in spoken and "54.3十亿美元" in spoken and "百分之12" in spoken and "基点" in spoken


@pytest.mark.parametrize("text,field",[("收入为 $999B","unsupported_money_base_units"),("Q2 FY2035 收入增长","unsupported_periods"),("收入同比下降 85%","direction_conflict"),("收入增长 999%","unsupported_percentages")])
def test_pre_written_conflict_blocks_multiple_claim_types(tmp_path:Path,text:str,field:str) -> None:
    """Money, period, direction, and percentage conflicts must all block."""

    req=sample(); req.transcript.mode="pre-written"; req.transcript.text=text
    pipeline=Pipeline("conflict",req,auto_approve=True,settings=local_settings(tmp_path)); facts=json.loads((ROOT/"examples/nvidia_q1_fy27/canonical_facts.json").read_text()); report=pipeline._conflict_check(facts)
    assert report and report[field]


def test_input_requires_report_disclaimer_and_generated_requirements() -> None:
    """The runtime model must enforce the public schema, not merely document it."""

    with pytest.raises(ValidationError): JobInput.model_validate({"source_materials":{"financial_reports":[],"disclaimer":""}})
    payload=sample().model_dump(mode="json"); payload["content_requirements"]=None
    with pytest.raises(ValidationError): JobInput.model_validate(payload)


def test_output_ratio_must_match_resolution() -> None:
    """Contradictory output geometry must fail before provider calls."""

    payload=sample().model_dump(mode="json"); payload["output"]={"aspect_ratio":"9:16","resolution":"1920x1080","fps":30,"format":"mp4"}
    with pytest.raises(ValidationError): JobInput.model_validate(payload)


def test_canonical_facts_reject_duplicate_ids() -> None:
    """Canonical fact references must be unambiguous."""

    data=json.loads((ROOT/"examples/nvidia_q1_fy27/canonical_facts.json").read_text()); data["facts"].append(data["facts"][0])
    with pytest.raises(ValidationError): CanonicalFacts.model_validate(data)


def test_planning_bundle_rejects_unknown_chart_type() -> None:
    """Provider output outside finite renderer capabilities must fail validation."""

    facts=json.loads((ROOT/"examples/nvidia_q1_fy27/canonical_facts.json").read_text()); payload={"canonical_facts":facts,"financial_analysis":{},"story_plan":{},"scene_plan":{"scenes":[{"id":"s1","purpose":"disclaimer","duration_seconds":4,"transition":"cut"}]},"narration":{"language":"zh-CN","segments":[{"scene_id":"s1","display_text":"x","spoken_text":"x","fact_ids":[]}]},"chart_spec":{"charts":[{"id":"x","type":"invented","animation":"grow"}]}}
    with pytest.raises(ValidationError): PlanningBundle.model_validate(payload)


def test_narration_rejects_unknown_fact_and_normalizes_spoken(tmp_path:Path) -> None:
    """Every narration fact ID must resolve and spoken text is deterministic."""

    pipeline=Pipeline("refs",sample(),settings=local_settings(tmp_path)); scenes=pipeline._scenes(); facts=json.loads((ROOT/"examples/nvidia_q1_fy27/canonical_facts.json").read_text()); narration=pipeline._narration(scenes); narration["segments"][0]["fact_ids"]=["missing"]
    with pytest.raises(ValueError,match="unknown facts"): pipeline._validate_narration(narration,scenes,facts)
    narration["segments"][0]["fact_ids"]=[]; narration["segments"][0]["spoken_text"]="BAD"; valid=pipeline._validate_narration(narration,scenes,facts)
    assert valid["segments"][0]["spoken_text"]==normalize_spoken(valid["segments"][0]["display_text"])


def test_chart_rejects_unknown_scene_chart_and_fact(tmp_path:Path) -> None:
    """Chart references must be valid before rendering."""

    pipeline=Pipeline("charts",sample(),settings=local_settings(tmp_path)); scenes=pipeline._scenes(); facts=json.loads((ROOT/"examples/nvidia_q1_fy27/canonical_facts.json").read_text()); charts=pipeline._charts(); scenes["scenes"][0]["chart"]="missing"
    with pytest.raises(ValueError,match="unknown chart"): pipeline._validate_charts(charts,scenes,facts)


def test_phrase_alignment_groups_character_timestamps(tmp_path:Path) -> None:
    """ElevenLabs character timestamps must become scene-linked phrases."""

    pipeline=Pipeline("align",sample(),settings=local_settings(tmp_path)); narration={"segments":[{"scene_id":"s1","display_text":"你好","spoken_text":"你好","fact_ids":[]},{"scene_id":"s2","display_text":"世界","spoken_text":"世界","fact_ids":[]}]}; raw={"character_start_times_seconds":[0,.2,.4,.6,.8],"character_end_times_seconds":[.2,.4,.6,.8,1.0]}; result=pipeline._phrase_alignment(narration,raw)
    assert [p["scene_id"] for p in result["phrases"]]==["s1","s2"] and result["duration_seconds"]>0


def test_review_gates_pause_and_resume_without_failure(tmp_path:Path,monkeypatch:pytest.MonkeyPatch) -> None:
    """Each approval resumes from durable artifacts and reaches the next gate."""

    monkeypatch.setattr(Pipeline,"_tts_audio",fake_tts); monkeypatch.setattr(Pipeline,"_render",fake_render); settings=local_settings(tmp_path); req=sample(); run=tmp_path/"review-job"
    record=Pipeline("review-job",req,settings=settings).run(); assert record.status==JobStatus.WAITING_REVIEW and record.current_node=="story_plan"
    approve(run,"story_plan"); record=Pipeline("review-job",req,settings=settings).run(); assert record.current_node=="narration"
    approve(run,"narration"); record=Pipeline("review-job",req,settings=settings).run(); assert record.current_node=="chart_spec"
    approve(run,"chart_spec"); record=Pipeline("review-job",req,settings=settings).run(); assert record.status==JobStatus.COMPLETED


def test_scene_retry_rebuilds_downstream_and_preserves_facts(tmp_path:Path,monkeypatch:pytest.MonkeyPatch) -> None:
    """Retry must update the selected scene and final video without changing facts."""

    monkeypatch.setattr(Pipeline,"_tts_audio",fake_tts); monkeypatch.setattr(Pipeline,"_render",fake_render); settings=local_settings(tmp_path); req=sample(); pipeline=Pipeline("retry-job",req,auto_approve=True,settings=settings); pipeline.run(); run=tmp_path/"retry-job"; before=(run/"canonical_facts.json").read_bytes(); retry_file=run/"retry_s2_1.json"; retry_file.write_text("{}")
    pipeline=Pipeline("retry-job",req,settings=settings); result=pipeline.retry_scene("s2",{"title":"Retried"},retry_file)
    assert result.status==JobStatus.COMPLETED and (run/"canonical_facts.json").read_bytes()==before and json.loads((run/"scene_plan.json").read_text())["scenes"][1]["title"]=="Retried" and json.loads(retry_file.read_text())["status"]=="completed"


def test_manifest_honors_output_brand_and_caption_settings(tmp_path:Path) -> None:
    """Public render settings must not be silently replaced by constants."""

    req=sample(); req.output.aspect_ratio="1:1"; req.output.resolution="1080x1080"; req.output.fps=60; req.brand.primary_color="#123456"; req.captions.max_lines=1; pipeline=Pipeline("manifest",req,settings=local_settings(tmp_path)); pipeline.run_dir.joinpath("tts.json").write_text("{}"); scenes=pipeline._scenes(); timeline=pipeline._timeline(scenes,{"phrases":[]}); manifest=pipeline._manifest(scenes,pipeline._narration(scenes),pipeline._charts(),timeline,pipeline._assets(scenes))
    assert manifest["output"]=={"width":1080,"height":1080,"fps":60,"codec":"h264"} and manifest["brand"]["primary_color"]=="#123456" and manifest["captions"]["max_lines"]==1


def test_qa_fails_when_video_is_missing(tmp_path:Path) -> None:
    """A workflow without an MP4 must never report successful QA."""

    pipeline=Pipeline("qa",sample(),settings=local_settings(tmp_path)); audio=ROOT/"public/qa-test.wav"; audio.write_bytes(b"audio"); pipeline.run_dir.joinpath("tts.json").write_text(json.dumps({"remotion_static_src":audio.name})); scenes=pipeline._scenes(); narration=pipeline._narration(scenes); charts=pipeline._charts(); manifest=pipeline._manifest(scenes,narration,charts,pipeline._timeline(scenes,pipeline._tts_alignment(narration)),pipeline._assets(scenes))
    with pytest.raises(RuntimeError,match="video_exists"): pipeline._qa(manifest)
    assert json.loads((pipeline.run_dir/"qa_report.json").read_text())["passed"] is False


def test_idempotency_deduplicates_and_rejects_key_reuse(tmp_path:Path,monkeypatch:pytest.MonkeyPatch) -> None:
    """One key maps to one request and accidental repeat clicks are free."""

    monkeypatch.setenv("ARTIFACT_ROOT",str(tmp_path)); monkeypatch.setenv("ADAPTER_MODE","deterministic"); client=TestClient(app); payload=sample().model_dump(mode="json"); headers={"Idempotency-Key":"same-key"}; first=client.post("/api/jobs",json=payload,headers=headers); second=client.post("/api/jobs",json=payload,headers=headers); payload["content_requirements"]["topic"]="changed"; conflict=client.post("/api/jobs",json=payload,headers=headers)
    assert first.status_code==202 and second.json()["deduplicated"] is True and conflict.status_code==409


def test_upload_rejects_non_pdf(tmp_path:Path,monkeypatch:pytest.MonkeyPatch) -> None:
    """The local upload boundary accepts only actual PDFs."""

    monkeypatch.setenv("ARTIFACT_ROOT",str(tmp_path)); response=TestClient(app).post("/api/uploads",files={"report":("fake.pdf",b"not pdf","application/pdf")})
    assert response.status_code==400


def test_metrics_replace_repeated_node_instead_of_double_counting(tmp_path:Path) -> None:
    """Resume operations must not double-count the same provider node."""

    pipeline=Pipeline("metrics",sample(),settings=local_settings(tmp_path)); pipeline._node("provider",{},usage_meta={"api_calls":1,"input_tokens":10}); pipeline._node("provider",{},usage_meta={"api_calls":1,"input_tokens":10}); metrics=json.loads((pipeline.run_dir/"metrics.json").read_text())
    assert metrics["totals"]["api_calls"]==1 and metrics["totals"]["input_tokens"]==10


def test_alignment_does_not_double_count_tts_characters(tmp_path:Path) -> None:
    """Character volume belongs to the TTS call, not its derived alignment artifact."""

    pipeline=Pipeline("characters",sample(),settings=local_settings(tmp_path)); pipeline._node("alignment",{}); pipeline._node("tts",{},tts_chars=123); metrics=json.loads((pipeline.run_dir/"metrics.json").read_text())
    assert metrics["totals"]["tts_characters"]==123


def test_production_configuration_fails_before_api_calls(monkeypatch:pytest.MonkeyPatch) -> None:
    """Production must never silently switch to deterministic providers."""

    for key in ("OPENAI_API_KEY","ELEVENLABS_API_KEY","ELEVENLABS_VOICE_ID","BACH_ACCESS_KEY","BACH_SECRET_KEY"): monkeypatch.delenv(key,raising=False)
    with pytest.raises(ValueError,match="OPENAI_API_KEY"): Settings.load("production").validate()


def test_elevenlabs_defaults_are_explicit() -> None:
    """The default narration model and output format remain stable and documented."""

    settings=Settings.load("deterministic")
    assert settings.elevenlabs_model_id=="eleven_multilingual_v2" and settings.elevenlabs_output_format=="mp3_44100_128"


def test_elevenlabs_request_matches_official_timestamp_contract(tmp_path:Path,monkeypatch:pytest.MonkeyPatch) -> None:
    """Output format is a query parameter while text and model remain in JSON."""

    captured={}
    class Response:
        """Minimal successful ElevenLabs HTTP response."""
        headers={"request-id":"test"}
        def raise_for_status(self) -> None: pass
        def json(self) -> dict: return {"audio_base64":base64.b64encode(b"mp3").decode(),"normalized_alignment":{"characters":["你"],"character_start_times_seconds":[0],"character_end_times_seconds":[.2]}}
    class Client:
        """Capture the outgoing request without using the network."""
        def __init__(self,**kwargs): captured["client_kwargs"]=kwargs
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def post(self,url,**kwargs): captured["url"]=url; captured.update(kwargs); return Response()
    monkeypatch.setattr("app.adapters.httpx.Client",Client); settings=replace(Settings.load("deterministic"),elevenlabs_api_key="secret",elevenlabs_voice_id="voice-123",elevenlabs_model_id="eleven_multilingual_v2",elevenlabs_output_format="mp3_44100_128"); output=tmp_path/"narration.mp3"; audio,meta=ElevenLabsTTS(settings).synthesize("你好",output)
    assert captured["url"].endswith("/text-to-speech/voice-123/with-timestamps")
    assert captured["headers"]["xi-api-key"]=="secret"
    assert captured["params"]=={"output_format":"mp3_44100_128"}
    assert captured["json"]=={"text":"你好","model_id":"eleven_multilingual_v2"}
    assert output.read_bytes()==b"mp3" and audio["alignment"]["characters"]==["你"] and meta["model_id"]=="eleven_multilingual_v2"


def test_bach_jwt_contains_official_claims(tmp_path:Path) -> None:
    """BACH authentication is an HS256 JWT whose issuer is the AccessKey."""

    settings=replace(local_settings(tmp_path),bach_access_key="ak-test",bach_secret_key="sk-test")
    token=BachAssets(settings)._token(now=1_000_000); header,claims,signature=token.split(".")
    decode=lambda value: json.loads(base64.urlsafe_b64decode(value+"="*(-len(value)%4)))
    assert decode(header)=={"alg":"HS256","typ":"JWT"}
    assert decode(claims)=={"iss":"ak-test","nbf":999_995,"exp":1_864_000}
    assert signature


def test_bach_text_to_image_matches_async_contract(tmp_path:Path,monkeypatch:pytest.MonkeyPatch) -> None:
    """Text-to-Image posts official fields and polls the same endpoint for image_urls."""

    captured=[]
    class Response:
        """Minimal BACH envelope response."""
        headers={"x-request-id":"bach-test"}
        def __init__(self,data): self.data=data
        def raise_for_status(self): pass
        def json(self): return {"code":200,"data":self.data,"timestamp":1}
    class Client:
        """Capture BACH submission and polling without network access."""
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def post(self,url,**kwargs): captured.append(("POST",url,kwargs)); return Response({"task_id":"task-1","status":"TASK_PENDING"})
        def get(self,url,**kwargs): captured.append(("GET",url,kwargs)); return Response({"task_id":"task-1","status":"TASK_SUCCEEDED","image_urls":["https://cdn.example/result.png"]})
    monkeypatch.setattr("app.adapters.httpx.Client",Client); monkeypatch.setattr("app.adapters.time.sleep",lambda _:None)
    settings=replace(local_settings(tmp_path),bach_access_key="ak",bach_secret_key="sk",bach_poll_interval_seconds=.001)
    result,meta=BachAssets(settings).text_to_image("cartoon finance scene")
    assert captured[0][1].endswith("/images/text2image") and captured[1][1].endswith("/images/text2image/task-1")
    assert captured[0][2]["json"]=={"prompt":"cartoon finance scene","output_count":1,"aspect_ratio":"16:9","image_size":"2K","quality":"medium","output_mime_type":"image/png"}
    assert captured[0][2]["headers"]["Authorization"].startswith("Bearer ") and result["image_urls"] and meta["api_calls"]==2


def test_bach_scene_routing_reuses_subject_and_skips_disclaimer(tmp_path:Path,monkeypatch:pytest.MonkeyPatch) -> None:
    """Repeated subjects use Elements; plain B-roll uses text; disclaimer stays deterministic."""

    calls=[]
    def subject(self,name,description,subject_type="character"):
        calls.append("subject"); return {"split_images":{str(i):f"https://cdn.example/{i}.png" for i in range(4)}},{"api_calls":1,"cost_usd":0}
    def elements(self,prompt,reference_inputs): calls.append("elements"); assert len(reference_inputs[0]["images"])==4; return {"video_url":"https://cdn.example/e.mp4"},{"api_calls":1,"cost_usd":0}
    def text_video(self,prompt): calls.append("text_video"); return {"video_url":"https://cdn.example/t.mp4"},{"api_calls":1,"cost_usd":0}
    monkeypatch.setattr(BachAssets,"text_to_subject",subject); monkeypatch.setattr(BachAssets,"elements_to_video",elements); monkeypatch.setattr(BachAssets,"text_to_video",text_video)
    pipeline=Pipeline("bach-routing",sample(),settings=replace(local_settings(tmp_path),mode="production",openai_api_key="openai",elevenlabs_api_key="eleven",elevenlabs_voice_id="voice",bach_access_key="ak",bach_secret_key="sk"))
    assets=pipeline._production_assets({"scenes":[{"id":"s1","purpose":"hook","asset_subject":"mascot"},{"id":"s2","purpose":"chart","asset_subject":"mascot"},{"id":"s3","purpose":"office"},{"id":"s4","purpose":"disclaimer"}]})
    assert calls==["subject","elements","elements","text_video"]
    assert assets["scenes"][-1]["strategy"]=="icon_illustration_gradient"
