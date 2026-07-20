"""Unit, integration, and contract tests for high-risk workflow behavior."""
from __future__ import annotations

import base64, json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.adapters import BachAssets, ElevenLabsTTS, OpenAIPlanner
from app.main import app
from app.models import CanonicalFacts, JobInput, JobStatus, PlanningBundle, ProviderPlanningBundle
from app.pipeline import Pipeline, ROOT, is_disclaimer_scene, money_base_units, normalize_spoken, percentage_claim_supported, visual_summary


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
    assert "$" not in spoken and "FY" not in spoken and "五百四十三亿美元" in spoken and "百分之十二" in spoken and "基点" in spoken
    assert normalize_spoken("$82B、$49B、$2.39/股、74.9%") == "八百二十亿美元、四百九十亿美元、每股二点三九美元、百分之七十四点九"


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


def test_openai_planning_schema_is_strict_outputs_compatible() -> None:
    """Every provider object is closed and requires every declared property."""

    schema=ProviderPlanningBundle.model_json_schema()
    def check(value:object) -> None:
        if isinstance(value,dict):
            assert "oneOf" not in value
            if value.get("type")=="object":
                assert value.get("additionalProperties") is False
                assert set(value.get("required",[]))==set(value.get("properties",{}))
            for child in value.values(): check(child)
        elif isinstance(value,list):
            for child in value: check(child)
    check(schema)


def test_openai_error_includes_provider_response(tmp_path:Path,monkeypatch:pytest.MonkeyPatch) -> None:
    """A rejected request exposes OpenAI's useful error body instead of a bare 400."""

    class Response:
        status_code=400
        text='{"error":{"message":"Invalid schema for response_format"}}'
        def raise_for_status(self) -> None:
            import httpx
            request=httpx.Request("POST","https://api.openai.com/v1/responses")
            response=httpx.Response(400,request=request)
            raise httpx.HTTPStatusError("bad request",request=request,response=response)
    class Client:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def post(self,*args,**kwargs): return Response()
    monkeypatch.setattr("app.adapters.httpx.Client",Client)
    settings=replace(local_settings(tmp_path),mode="production",openai_api_key="test")
    with pytest.raises(RuntimeError,match="Invalid schema for response_format"):
        OpenAIPlanner(settings).generate({}, {})


def test_provider_bundle_compiles_numbers_from_fact_references() -> None:
    """Narration and charts derive numeric output from normalized facts, not model text."""

    payload={
        "canonical_facts":{"schema_version":"1","entity":"Example Co","ticker":"EX","report":{"title":"Q1 report","document_type":"earnings release","fiscal_period":"Q1 FY27","period_end":"2026-04-26"},"facts":[{"id":"revenue","metric":"Revenue","quantity":{"kind":"money","amount":81615,"currency":"USD","magnitude":"millions"},"reported_value":81615,"reported_unit_text":"USD millions","basis":"GAAP","fiscal_period":"Q1 FY27","period_end":"2026-04-26","source":"report.pdf","source_locator":"page 1","confidence":.95,"derived_from":[],"formula":None}]},
        "financial_analysis":{"summary":"record revenue","insights":[{"title":"Revenue","summary":"record result","fact_ids":["revenue"]}]},
        "story_plan":{"title":"Results","thesis":"Growth","beats":[{"purpose":"result","summary":"revenue","fact_ids":["revenue"]}]},
        "scene_plan":[{"id":"s1","kind":"content","purpose":"result","duration_seconds":10,"title":"Revenue","chart":"c1","asset_subject":None,"subject_id":None,"visual_prompt":None,"transition":"cut"},{"id":"s2","kind":"disclaimer","purpose":"结尾法律提示","duration_seconds":5,"title":"免责声明","chart":None,"asset_subject":None,"subject_id":None,"visual_prompt":None,"transition":"cut"}],
        "narration":{"language":"zh-CN","segments":[{"scene_id":"s1","parts":[{"type":"text","value":"营收达到"},{"type":"fact","fact_id":"revenue","precision":1,"compact":True}]},{"scene_id":"s2","parts":[{"type":"text","value":"本视频不构成投资建议"}]}],"source":"generated","editing_applied":False},
        "chart_spec":{"caption_region":{"x":96,"y":756,"width":1728,"height":216},"charts":[{"id":"c1","type":"bar","title":"Revenue","series":[{"label":"Q1 FY27","fact_id":"revenue","role":"value"}],"precision":1,"compact":True,"animation":"grow"}]},
    }
    content=[]; narration=[]; charts=[]
    for index,kind in enumerate(["chart","chart","chart","metric_cards","broll"],1):
        chart_id=f"c{index}" if kind!="broll" else None
        content.append({**payload["scene_plan"][0],"id":f"s{index}","visual_kind":kind,"chart":chart_id})
        narration.append({**payload["narration"]["segments"][0],"scene_id":f"s{index}"})
        if chart_id: charts.append({**payload["chart_spec"]["charts"][0],"id":chart_id,"type":"metric_cards" if kind=="metric_cards" else "bar"})
    payload["scene_plan"]=content+[{**payload["scene_plan"][1],"id":"s6","visual_kind":"disclaimer"}]
    payload["scene_plan"][3]["visual_kind"]="chart"  # provider label conflicts with c4 metric_cards
    payload["narration"]["segments"]=narration+[{**payload["narration"]["segments"][1],"scene_id":"s6"}]
    payload["chart_spec"]["charts"]=charts
    result=ProviderPlanningBundle.model_validate(payload).to_domain().model_dump(mode="json")
    fact=result["canonical_facts"]["facts"][0]
    assert fact["value"]==81_615_000_000 and fact["scale"]=="ones" and fact["reported"]["value"]==81615
    assert result["narration"]["segments"][0]["display_text"]=="营收达到$81.6B"
    assert result["chart_spec"]["charts"][0]["values"]==[81.6]
    assert result["scene_plan"]["scenes"][3]["visual_kind"]=="metric_cards"


def test_provider_text_parts_cannot_smuggle_numeric_claims() -> None:
    """All model-authored numbers must enter narration through fact parts."""

    from app.models import ProviderTextPart
    with pytest.raises(ValidationError): ProviderTextPart.model_validate({"type":"text","value":"营收 $99B"})


def test_disclaimer_kind_is_language_independent() -> None:
    """Program logic uses stable kind while still recognizing legacy Chinese tasks."""

    assert is_disclaimer_scene({"kind":"disclaimer","purpose":"legal close"})
    assert is_disclaimer_scene({"purpose":"结尾免责声明与法律提示"})
    assert not is_disclaimer_scene({"kind":"content","purpose":"earnings summary"})


def test_generated_narration_uses_request_disclaimer(tmp_path:Path) -> None:
    """Planner-authored disclaimer text must never override the submitted form value."""

    request=sample(); request.source_materials.disclaimer="这是前端提交的专属免责声明。"
    pipeline=Pipeline("generated-disclaimer",request,settings=local_settings(tmp_path)); scenes=pipeline._scenes(); generated=pipeline._narration(scenes)
    generated["segments"][-1]["display_text"]="模型生成的旧免责声明"
    generated["segments"][-1]["spoken_text"]="模型生成的旧免责声明"
    resolved=pipeline._resolve_narration(scenes,{"narration":generated})
    disclaimer_segment=next(segment for segment in resolved["segments"] if segment["scene_id"]=="s6")
    assert disclaimer_segment["display_text"]==request.source_materials.disclaimer
    assert disclaimer_segment["spoken_text"]==normalize_spoken(request.source_materials.disclaimer)
    assert disclaimer_segment["fact_ids"]==[]


def test_narration_rejects_unknown_fact_and_normalizes_spoken(tmp_path:Path) -> None:
    """Every narration fact ID must resolve and spoken text is deterministic."""

    pipeline=Pipeline("refs",sample(),settings=local_settings(tmp_path)); scenes=pipeline._scenes(); facts=json.loads((ROOT/"examples/nvidia_q1_fy27/canonical_facts.json").read_text()); narration=pipeline._narration(scenes); narration["segments"][0]["fact_ids"]=["missing"]
    with pytest.raises(ValueError,match="unknown facts"): pipeline._validate_narration(narration,scenes,facts)
    narration["segments"][0]["fact_ids"]=[]; narration["segments"][0]["spoken_text"]="BAD"; valid=pipeline._validate_narration(narration,scenes,facts)
    assert valid["segments"][0]["spoken_text"]==normalize_spoken(valid["segments"][0]["display_text"])


def test_scaled_currency_fact_matches_abbreviated_narration(tmp_path:Path) -> None:
    """A report value expressed in millions must validate against a $B caption."""

    fact={"id":"revenue","metric":"Revenue","value":81615,"unit":"USD","scale":"millions","currency":"USD","basis":"GAAP","fiscal_period":"Q1 FY27","period_end":"2026-04-26","comparison":{},"source":"report","source_locator":"page 1","confidence":.9,"derived_from":[],"formula":None}
    assert money_base_units(fact)==81_615_000_000
    narration={"language":"zh-CN","segments":[{"scene_id":"scene2","display_text":"营收为 $81.6B","spoken_text":"营收为八百一十六亿美元","fact_ids":["revenue"]}],"source":"generated","editing_applied":False}
    scenes={"scenes":[{"id":"scene2","purpose":"financial results","duration_seconds":10,"transition":"cut"}]}
    result=Pipeline("scaled-money",sample(),settings=local_settings(tmp_path))._validate_narration(narration,scenes,{"facts":[fact]})
    assert result["segments"][0]["scene_id"]=="scene2"


def test_percentage_validation_respects_display_precision(tmp_path:Path) -> None:
    """A whole percentage may round 74.9 to 75 without allowing unrelated claims."""

    assert percentage_claim_supported("75",{74.9})
    assert percentage_claim_supported("74.9",{74.9})
    assert not percentage_claim_supported("74",{74.9})
    fact={"id":"margin","metric":"Gross margin","value":74.9,"unit":"%","scale":"","currency":None,"basis":"GAAP","fiscal_period":"Q1 FY27","period_end":"2026-04-26","comparison":{},"source":"report","source_locator":"page 1","confidence":.9,"derived_from":[],"formula":None}
    narration={"language":"zh-CN","segments":[{"scene_id":"scene3","display_text":"毛利率接近75%","spoken_text":"毛利率接近百分之七十五","fact_ids":["margin"]}],"source":"generated","editing_applied":False}
    scenes={"scenes":[{"id":"scene3","purpose":"margin","duration_seconds":10,"transition":"cut"}]}
    Pipeline("rounded-percent",sample(),settings=local_settings(tmp_path))._validate_narration(narration,scenes,{"facts":[fact]})


def test_chart_rejects_unknown_scene_chart_and_fact(tmp_path:Path) -> None:
    """Chart references must be valid before rendering."""

    pipeline=Pipeline("charts",sample(),settings=local_settings(tmp_path)); scenes=pipeline._scenes(); facts=json.loads((ROOT/"examples/nvidia_q1_fy27/canonical_facts.json").read_text()); charts=pipeline._charts(); scenes["scenes"][0]["chart"]="missing"
    with pytest.raises(ValueError,match="unknown chart"): pipeline._validate_charts(charts,scenes,facts)


def test_phrase_alignment_groups_character_timestamps(tmp_path:Path) -> None:
    """ElevenLabs character timestamps must become scene-linked phrases."""

    pipeline=Pipeline("align",sample(),settings=local_settings(tmp_path)); narration={"segments":[{"scene_id":"s1","display_text":"你好","spoken_text":"你好","fact_ids":[]},{"scene_id":"s2","display_text":"世界","spoken_text":"世界","fact_ids":[]}]}; raw={"character_start_times_seconds":[0,.2,.4,.6,.8],"character_end_times_seconds":[.2,.4,.6,.8,1.0]}; result=pipeline._phrase_alignment(narration,raw)
    assert [p["scene_id"] for p in result["phrases"]]==["s1","s2"] and result["duration_seconds"]>0


def test_normalized_pinyin_alignment_uses_full_audio_duration(tmp_path:Path) -> None:
    """Expanded Mandarin pinyin indices must not collapse all cues into the audio opening."""

    pipeline=Pipeline("pinyin-align",sample(),settings=local_settings(tmp_path)); narration={"segments":[{"scene_id":"s1","display_text":"你好。","spoken_text":"你好。","fact_ids":[]},{"scene_id":"s2","display_text":"世界。","spoken_text":"世界。","fact_ids":[]}]}; _,spans=pipeline._tts_script(narration)
    count=28; raw={"characters":list("x"*count),"character_start_times_seconds":[i*.5 for i in range(count)],"character_end_times_seconds":[(i+1)*.5 for i in range(count)]}
    result=pipeline._phrase_alignment(spans,raw)
    assert result["alignment_index_mode"]=="proportional_fallback"
    assert result["caption_cues"][-1]["end_seconds"]==14 and result["duration_seconds"]==14


def test_audio_driven_timeline_does_not_stretch_short_audio(tmp_path:Path) -> None:
    """A 19-second voice track must not be spread across planned 70-second scenes."""

    pipeline=Pipeline("audio-clock",sample(),settings=local_settings(tmp_path)); scenes=pipeline._scenes(); narration=pipeline._narration(scenes); alignment=pipeline._tts_alignment(narration)
    scale=19/alignment["duration_seconds"]
    for cue in alignment["caption_cues"]:
        cue["start_seconds"]*=scale; cue["end_seconds"]*=scale
    for phrase in alignment["phrases"]:
        phrase["start_seconds"]*=scale; phrase["end_seconds"]*=scale
    alignment["duration_seconds"]=19
    timeline=pipeline._timeline(scenes,alignment)
    assert 19 <= timeline["video_duration_seconds"] <= 23
    assert all(cue["from_frame"]<=cue["to_frame"] for cue in timeline["caption_cues"])


def test_default_plan_is_data_first(tmp_path:Path) -> None:
    """The built-in six-scene plan keeps B-roll to one scene and data dominant."""

    pipeline=Pipeline("visual-mix",sample(),settings=local_settings(tmp_path)); scenes=pipeline._scenes(); pipeline._validate_visual_mix(scenes)
    summary=visual_summary(scenes)
    assert summary["data_visual_coverage"]==.8 and summary["chart_count"]>=2 and summary["metric_card_count"]==1 and summary["broll_count"]==1


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


def test_tts_script_keeps_decimal_caption_clauses_short(tmp_path:Path) -> None:
    """Decimal points must not make display/spoken clause counts diverge."""

    pipeline=Pipeline("caption-decimal",sample(),settings=local_settings(tmp_path))
    narration={"segments":[{"scene_id":"s1","display_text":"营收$81.6B，增长85%。","spoken_text":"营收八百一十六亿美元，增长百分之八十五。"}]}
    _,spans=pipeline._tts_script(narration)
    assert [span["text"] for span in spans]==["营收$81.6B，","增长85%。"]


def test_production_configuration_fails_before_api_calls(monkeypatch:pytest.MonkeyPatch) -> None:
    """Production must never silently switch to deterministic providers."""

    for key in ("OPENAI_API_KEY","ELEVENLABS_API_KEY","ELEVENLABS_VOICE_ID","BACH_ACCESS_KEY","BACH_SECRET_KEY"): monkeypatch.setenv(key,"")
    with pytest.raises(ValueError,match="OPENAI_API_KEY"): Settings.load("production").validate()


def test_elevenlabs_defaults_are_explicit(monkeypatch:pytest.MonkeyPatch) -> None:
    """The default narration model and output format remain stable and documented."""

    monkeypatch.setenv("ELEVENLABS_SPEED","1.02"); monkeypatch.setenv("MANDARIN_CHARS_PER_SECOND","4.4")
    monkeypatch.setenv("NARRATION_PLANNING_TOLERANCE","0.10"); monkeypatch.setenv("TARGET_DURATION_TOLERANCE","0.15")
    monkeypatch.setenv("NARRATION_REWRITE_ATTEMPTS","3"); monkeypatch.setenv("TTS_AUTO_FIT_DURATION","true")
    settings=Settings.load("deterministic")
    assert settings.elevenlabs_model_id=="eleven_multilingual_v2" and settings.elevenlabs_output_format=="mp3_44100_128"
    assert settings.elevenlabs_speed==1.02 and settings.mandarin_chars_per_second==4.4
    assert settings.narration_planning_tolerance==.1 and settings.target_duration_tolerance==.15
    assert settings.narration_rewrite_attempts==3 and settings.tts_auto_fit_duration is True


def test_generated_audio_duration_must_match_target(tmp_path:Path) -> None:
    """Large actual TTS drift must stop before expensive visual generation."""

    pipeline=Pipeline("duration-check",sample(),settings=local_settings(tmp_path))
    pipeline._validate_target_duration({"duration_seconds":70})
    with pytest.raises(RuntimeError,match="outside target"):
        pipeline._validate_target_duration({"duration_seconds":112.571})
    pipeline.request.transcript.mode="pre-written"; pipeline.request.transcript.text="用户锁定文案"
    with pytest.raises(RuntimeError,match="outside target"):
        pipeline._validate_target_duration({"duration_seconds":112.571})


def test_tts_speed_auto_fits_real_duration(tmp_path:Path) -> None:
    """A moderate Voice-specific duration miss should trigger one bounded speed correction."""

    request=sample(); request.content_requirements.target_duration_seconds=90
    settings=replace(local_settings(tmp_path),target_duration_tolerance=.15)
    pipeline=Pipeline("duration-fit",request,settings=settings)
    assert pipeline._duration_adjusted_speed(105.14,1.02)==1.192
    assert pipeline._duration_adjusted_speed(103,1.02) is None
    assert pipeline._duration_adjusted_speed(95,1.02) is None
    assert pipeline._duration_adjusted_speed(140,1.02)==1.2
    pipeline.request.transcript.mode="pre-written"; pipeline.request.transcript.text="用户锁定文案"; pipeline.request.transcript.allow_editing=False
    assert pipeline._duration_adjusted_speed(105.14,1.02)==1.192


def test_second_tts_failure_reports_fit_details(tmp_path:Path) -> None:
    """A final duration failure must report both TTS passes and the adjusted speed."""

    request=sample(); request.content_requirements.target_duration_seconds=90
    pipeline=Pipeline("duration-error",request,settings=replace(local_settings(tmp_path),target_duration_tolerance=.15))
    fit={"initial_duration_seconds":105.1,"initial_speed":1.02,"adjusted_speed":1.192,"adjusted_duration_seconds":104.0}
    with pytest.raises(RuntimeError,match=r"initial=105.1s, adjusted_speed=1.192, second=104.0s"):
        pipeline._validate_target_duration({"duration_seconds":104.0},fit)


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
    assert captured["json"]["text"]=="你好" and captured["json"]["model_id"]=="eleven_multilingual_v2"
    assert captured["json"]["voice_settings"]=={"stability":.45,"similarity_boost":.75,"style":0,"use_speaker_boost":True,"speed":.95}
    assert output.read_bytes()==b"mp3" and audio["alignment"]["characters"]==["你"] and meta["model_id"]=="eleven_multilingual_v2"


def test_elevenlabs_job_voice_overrides_environment_default(tmp_path:Path,monkeypatch:pytest.MonkeyPatch) -> None:
    """A per-job Voice ID must reach the provider URL and persisted metadata."""

    captured={}
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"audio_base64":base64.b64encode(b"mp3").decode(),"alignment":{}}
    class Client:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def post(self,url,**kwargs): captured["url"]=url; return Response()
    monkeypatch.setattr("app.adapters.httpx.Client",Client)
    settings=replace(local_settings(tmp_path),elevenlabs_voice_id="environment-voice")
    audio,meta=ElevenLabsTTS(settings).synthesize("你好",tmp_path/"voice.mp3",voice_id="job-voice")
    assert "/job-voice/" in captured["url"] and audio["voice_id"]=="job-voice" and meta["voice_id"]=="job-voice"


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
    def elements(self,prompt,reference_inputs,duration=6): calls.append("elements"); assert len(reference_inputs[0]["images"])==4 and 6<=duration<=10; return {"video_url":"https://cdn.example/e.mp4"},{"api_calls":1,"cost_usd":0}
    def text_video(self,prompt,duration=6): calls.append("text_video"); assert 6<=duration<=10; return {"video_url":"https://cdn.example/t.mp4"},{"api_calls":1,"cost_usd":0}
    monkeypatch.setattr(BachAssets,"text_to_subject",subject); monkeypatch.setattr(BachAssets,"elements_to_video",elements); monkeypatch.setattr(BachAssets,"text_to_video",text_video)
    pipeline=Pipeline("bach-routing",sample(),settings=replace(local_settings(tmp_path),mode="production",openai_api_key="openai",elevenlabs_api_key="eleven",elevenlabs_voice_id="voice",bach_access_key="ak",bach_secret_key="sk"))
    assets=pipeline._production_assets({"scenes":[{"id":"s1","purpose":"hook","asset_subject":"mascot"},{"id":"s2","purpose":"chart","asset_subject":"mascot"},{"id":"s3","purpose":"office"},{"id":"s4","purpose":"disclaimer"}]})
    assert calls==["subject","elements","elements","text_video"]
    assert next(item for item in assets["scenes"] if item["scene_id"]=="s4")["strategy"]=="disclaimer_template"
