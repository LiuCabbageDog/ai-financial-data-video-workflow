"""Deterministic, artifact-first workflow used locally and by the API worker."""
from __future__ import annotations

import hashlib, json, re, shutil, subprocess, time
from pathlib import Path
from typing import Any

from .models import CanonicalFacts, ChartSpec, JobInput, JobRecord, JobStatus, Narration, ScenePlan
from .observability import RunLogger
from .config import Settings
from .adapters import BachAssets, ElevenLabsTTS, OpenAIPlanner, parse_financial_reports

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REVIEW_NODES = {"story_plan", "narration", "chart_spec"}
TRANSITIONS = {"cut", "fade", "slide", "wipe", "zoom"}


def canonical_json(value: Any) -> str:
    """Serialize stable JSON for idempotency and cache hashing."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    """Return a SHA-256 digest of normalized content."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def normalize_spoken(display: str) -> str:
    """Convert common financial notation to stable Mandarin TTS text."""
    replacements = [(r"Data Center", "数据中心"),(r"non[- ]?GAAP", "非美国通用会计准则"),(r"GAAP", "美国通用会计准则"),(r"bps\b", "个基点")]
    spoken=display
    for pattern, value in replacements: spoken=re.sub(pattern,value,spoken,flags=re.I)
    spoken=re.sub(r"Q([1-4])\s*FY(\d{4})",lambda m:f"{m.group(2)}财年第{m.group(1)}季度",spoken,flags=re.I)
    scales={"K":"千","M":"百万","B":"十亿","T":"万亿"}
    spoken=re.sub(r"([$¥€£])\s*(-?\d+(?:\.\d+)?)\s*([KMBT])\b",lambda m:f"{m.group(2)}{scales[m.group(3).upper()]}{'美元' if m.group(1)=='$' else '元'}",spoken,flags=re.I)
    spoken=re.sub(r"(\d+(?:\.\d+)?)%", lambda m: f"百分之{m.group(1)}", spoken)
    return spoken.replace("+", "增长").replace("±", "上下浮动").replace("−","负").replace("-", "负")


def write_artifact(run_dir: Path, name: str, value: Any) -> str:
    """Write a named immutable-style JSON artifact and return its filename."""
    path=run_dir/name; path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8"); return name


class Pipeline:
    """Orchestrate validated nodes while preserving every intermediate output."""
    def __init__(self, job_id: str, request: JobInput, auto_approve: bool=False, settings: Settings | None=None):
        self.job_id, self.request, self.auto_approve = job_id, request, auto_approve
        self.settings=settings or Settings.load(); self.settings.validate()
        self.run_dir=self.settings.artifact_root/job_id; self.run_dir.mkdir(parents=True,exist_ok=True)
        self.logger=RunLogger(self.run_dir,job_id)
        record_path=self.run_dir/"job.json"
        self.record=JobRecord.model_validate_json(record_path.read_text()) if record_path.exists() else JobRecord(id=job_id,status=JobStatus.QUEUED)

    def save_record(self) -> None:
        """Atomically-enough persist current task status for local reads."""
        (self.run_dir/"job.json").write_text(self.record.model_dump_json(indent=2),encoding="utf-8")

    def retry_scene(self,scene_id:str,overrides:dict[str,Any],retry_path:Path)->JobRecord:
        """Regenerate one scene's assets and rebuild only shared downstream artifacts."""
        allowed={"purpose","title","duration_seconds","visual_prompt","transition","chart","asset_subject","subject_id"}
        unknown=set(overrides)-allowed
        if unknown: raise ValueError(f"unsupported scene retry overrides: {sorted(unknown)}")
        self.record.status=JobStatus.RUNNING; self.record.current_node=f"retry:{scene_id}"; self.save_record(); started=time.time()
        try:
            scenes=json.loads((self.run_dir/"scene_plan.json").read_text()); target=next(s for s in scenes["scenes"] if s["id"]==scene_id); target.update(overrides); ScenePlan.model_validate(scenes); self._node("scene_plan",scenes)
            existing=json.loads((self.run_dir/"assets.json").read_text()); single={"scenes":[target]}; generated=self._production_assets(single) if self.settings.mode=="production" else self._assets(single)
            existing["scenes"]=[s for s in existing["scenes"] if s["scene_id"]!=scene_id]+generated["scenes"]; existing.setdefault("reference_registry",{}).update(generated.get("reference_registry",{})); self._node("assets",existing)
            narration=json.loads((self.run_dir/"narration.json").read_text()); charts=json.loads((self.run_dir/"chart_spec.json").read_text()); alignment=json.loads((self.run_dir/"alignment.json").read_text()); timeline=self._timeline(scenes,alignment); self._node("animation_timeline",timeline)
            manifest=self._manifest(scenes,narration,charts,timeline,existing); self._node("render_manifest",manifest); self.record.status=JobStatus.RENDERING; self.save_record(); self._render(); self.record.status=JobStatus.QA; self.save_record(); self._qa(manifest)
            self.record.status=JobStatus.COMPLETED; self.record.current_node=None; self.record.progress=1; self.logger.metrics["totals"]["retries"]+=1; self.logger.metrics_path.write_text(json.dumps(self.logger.metrics,ensure_ascii=False,indent=2)); self.save_record(); retry_path.write_text(json.dumps({"scene_id":scene_id,"overrides":overrides,"status":"completed","started_at":started,"ended_at":time.time()},ensure_ascii=False,indent=2)); return self.record
        except Exception as exc:
            self.record.status=JobStatus.FAILED; self.record.error=f"{type(exc).__name__}: {exc}"; self.save_record(); retry_path.write_text(json.dumps({"scene_id":scene_id,"overrides":overrides,"status":"failed","reason":self.record.error,"started_at":started,"ended_at":time.time()},ensure_ascii=False,indent=2)); raise

    def run(self) -> JobRecord:
        """Execute the full deterministic demo and stop cleanly on conflicts/reviews."""
        self.record.status=JobStatus.RUNNING; self.save_record(); self.logger.event("workflow.started", input_hash=content_hash(self.request.model_dump()))
        try:
            self._node("input", self.request.model_dump())
            if self.settings.mode == "production":
                bundle_path=self.run_dir/"planning_bundle.json"
                if bundle_path.exists(): bundle=json.loads(bundle_path.read_text())
                else:
                    parsed=parse_financial_reports(self.request.source_materials.financial_reports); self._node("parsed_reports",parsed)
                    bundle,provider_meta=OpenAIPlanner(self.settings).generate(self.request.model_dump(mode="json"),parsed); self._node("planning_provider",provider_meta,usage_meta=provider_meta); self._node("planning_bundle",bundle)
                facts=bundle["canonical_facts"]
            else:
                bundle=None; facts=json.loads((ROOT/"examples/nvidia_q1_fy27/canonical_facts.json").read_text())
            facts=CanonicalFacts.model_validate(facts).model_dump(mode="json"); self._node("canonical_facts", facts)
            conflict=self._conflict_check(facts)
            if conflict:
                self._node("conflict_report", conflict); self.record.status=JobStatus.BLOCKED_CONFLICT; self.save_record(); return self.record
            analysis=bundle["financial_analysis"] if bundle else self._analysis(facts); self._node("financial_analysis",analysis)
            story=bundle["story_plan"] if bundle else self._story()
            if not self._reviewable("story_plan",story): return self.record
            scenes=ScenePlan.model_validate(bundle["scene_plan"] if bundle else self._scenes()).model_dump(mode="json"); self._node("scene_plan",scenes)
            narration=self._validate_narration(self._resolve_narration(scenes,bundle),scenes,facts)
            if not self._reviewable("narration",narration): return self.record
            charts=ChartSpec.model_validate(bundle["chart_spec"] if bundle else self._charts()).model_dump(mode="json"); self._validate_charts(charts,scenes,facts)
            if not self._reviewable("chart_spec",charts): return self.record
            if self.settings.mode == "production":
                spoken="。".join(x["spoken_text"] for x in narration["segments"]); public=ROOT/"public"; public.mkdir(exist_ok=True); audio_name=f"{self.job_id}-narration.mp3"
                audio,tts_meta=ElevenLabsTTS(self.settings).synthesize(spoken,public/audio_name); shutil.copy2(public/audio_name,self.run_dir/"narration.mp3"); audio["remotion_static_src"]=audio_name
                alignment=self._phrase_alignment(narration,audio.pop("alignment")); self._node("alignment",alignment)
                self._node("tts",audio,tts_chars=len(spoken),usage_meta=tts_meta)
            else:
                alignment=self._tts_alignment(narration); self._node("alignment",alignment)
                audio=self._tts_audio(narration); self._node("tts",audio,tts_chars=sum(len(x["spoken_text"]) for x in narration["segments"]))
            timeline=self._timeline(scenes,alignment); self._node("animation_timeline",timeline)
            assets=self._production_assets(scenes) if self.settings.mode=="production" else self._assets(scenes); self._node("assets",assets,usage_meta=assets.get("usage"))
            manifest=self._manifest(scenes,narration,charts,timeline,assets); self._node("render_manifest",manifest)
            self.record.status=JobStatus.RENDERING; self.save_record(); self._render()
            self.record.status=JobStatus.QA; self.save_record(); self._qa(manifest)
            self.record.status=JobStatus.COMPLETED; self.record.current_node=None; self.record.progress=1; self.logger.event("workflow.completed",artifacts=self.record.artifacts,metrics=self.logger.metrics.get("totals")); self.save_record(); return self.record
        except Exception as exc:
            self.record.status=JobStatus.FAILED; self.record.error=f"{type(exc).__name__}: {exc}"; self.logger.event("workflow.failed",reason=self.record.error); self.save_record(); raise

    def _node(self,name:str,value:Any,tts_chars:int=0,usage_meta:dict[str,Any]|None=None) -> None:
        """Persist one artifact and update node status/metrics."""
        with self.logger.node(name) as usage:
            filename=write_artifact(self.run_dir,f"{name}.json",value); usage["outputs"]=[filename]; usage["tts_characters"]=tts_chars
            for key in ("api_calls","input_tokens","output_tokens","cost_usd","latency_ms"):
                if usage_meta and key in usage_meta: usage[key]=usage_meta[key]
        self.record.nodes[name]="completed"
        if filename not in self.record.artifacts: self.record.artifacts.append(filename)
        self.record.current_node=name; self.record.progress=min(.9,len(self.record.nodes)/18); self.save_record()

    def _reviewable(self,name:str,value:Any) -> bool:
        """Persist a resumable review gate and return whether execution may continue."""
        self._node(name,value); path=self.run_dir/f"review_{name}.json"
        review=json.loads(path.read_text()) if path.exists() else {"node":name,"status":"approved" if self.auto_approve else "pending","reviewer":"cli-auto-approve" if self.auto_approve else None,"at":time.time() if self.auto_approve else None}
        write_artifact(self.run_dir,path.name,review)
        if review.get("status")!="approved":
            self.record.status=JobStatus.WAITING_REVIEW; self.record.current_node=name; self.record.nodes[name]="waiting_review"; self.save_record(); self.logger.event("review.waiting",node_id=name); return False
        self.record.nodes[name]="approved"; self.save_record(); return True

    def _conflict_check(self,facts:dict[str,Any]) -> dict[str,Any]|None:
        """Block unsupported percentage, money, fiscal-period, or direction claims."""
        if self.request.transcript.mode!="pre-written": return None
        text=self.request.transcript.text or ""; fact_list=facts["facts"]
        allowed_pct={round(float(v),4) for f in fact_list for v in f.get("comparison",{}).values()}|{round(float(f["value"]),4) for f in fact_list if f.get("unit")=="percent"}
        claimed_pct={round(float(x),4) for x in re.findall(r"(-?\d+(?:\.\d+)?)\s*%",text)}
        scales={"K":1e3,"M":1e6,"B":1e9,"T":1e12}; claimed_money=[]
        for amount,scale in re.findall(r"[$¥€£]\s*(-?\d+(?:\.\d+)?)\s*([KMBT])\b",text,re.I): claimed_money.append(float(amount)*scales[scale.upper()])
        allowed_money=[float(f["value"]) for f in fact_list if f.get("currency")]
        bad_money=[v for v in claimed_money if not any(abs(v-a)<=max(1,abs(a)*.006) for a in allowed_money)]
        periods=set(re.findall(r"Q[1-4]\s*FY\d{4}",text,re.I)); allowed_periods={str(f.get("fiscal_period","")).replace(" ","").upper() for f in fact_list}
        bad_periods=sorted(p for p in periods if p.replace(" ","").upper() not in allowed_periods)
        direction_error=bool(re.search(r"(?:下降|减少|down|declin)",text,re.I) and any(v>0 for v in claimed_pct) and claimed_pct & {abs(v) for v in allowed_pct if v>0})
        report={"status":"blocked","reason":"transcript_fact_conflict","unsupported_percentages":sorted(claimed_pct-allowed_pct),"unsupported_money_base_units":bad_money,"unsupported_periods":bad_periods,"direction_conflict":direction_error,"action":"Correct transcript or provide a cited supplementary source."}
        return report if any([report["unsupported_percentages"],bad_money,bad_periods,direction_error]) else None

    def _validate_narration(self,narration:dict[str,Any],scenes:dict[str,Any],facts:dict[str,Any])->dict[str,Any]:
        """Normalize spoken text and enforce scene/fact referential integrity."""
        narration=Narration.model_validate(narration).model_dump(mode="json"); scene_ids={s["id"] for s in scenes["scenes"]}; fact_ids={f["id"] for f in facts["facts"]}
        for segment in narration["segments"]:
            if segment["scene_id"] not in scene_ids: raise ValueError(f"narration references unknown scene: {segment['scene_id']}")
            missing=set(segment["fact_ids"])-fact_ids
            if missing: raise ValueError(f"narration references unknown facts: {sorted(missing)}")
            referenced=[f for f in facts["facts"] if f["id"] in segment["fact_ids"]]
            claimed_pct={float(v) for v in re.findall(r"(-?\d+(?:\.\d+)?)\s*%",segment["display_text"])}; allowed_pct={float(v) for f in referenced for v in f.get("comparison",{}).values()}|{float(f["value"]) for f in referenced if f.get("unit")=="percent"}
            if any(not any(abs(v-a)<.01 for a in allowed_pct) for v in claimed_pct): raise ValueError(f"narration contains unsupported percentage in scene {segment['scene_id']}")
            scales={"K":1e3,"M":1e6,"B":1e9,"T":1e12}; claimed_money=[float(v)*scales[s.upper()] for v,s in re.findall(r"[$¥€£]\s*(-?\d+(?:\.\d+)?)\s*([KMBT])\b",segment["display_text"],re.I)]; allowed_money=[float(f["value"]) for f in referenced if f.get("currency")]
            if any(not any(abs(v-a)<=max(1,abs(a)*.006) for a in allowed_money) for v in claimed_money): raise ValueError(f"narration contains unsupported money value in scene {segment['scene_id']}")
            segment["spoken_text"]=normalize_spoken(segment["display_text"])
        return narration

    def _validate_charts(self,charts:dict[str,Any],scenes:dict[str,Any],facts:dict[str,Any])->None:
        """Reject unknown chart and fact references before rendering."""
        chart_ids={c["id"] for c in charts["charts"]}; fact_ids={f["id"] for f in facts["facts"]}
        for scene in scenes["scenes"]:
            if scene.get("chart") and scene["chart"] not in chart_ids: raise ValueError(f"scene references unknown chart: {scene['chart']}")
        for chart in charts["charts"]:
            missing=set(chart.get("fact_ids",[]))-fact_ids
            if missing: raise ValueError(f"chart references unknown facts: {sorted(missing)}")

    def _phrase_alignment(self,narration:dict[str,Any],raw:dict[str,Any])->dict[str,Any]:
        """Group ElevenLabs character timestamps into scene-level phrases."""
        starts=raw.get("character_start_times_seconds",[]); ends=raw.get("character_end_times_seconds",[]); cursor=0; phrases=[]
        for segment in narration["segments"]:
            length=len(segment["spoken_text"]); start=starts[cursor] if cursor<len(starts) else (phrases[-1]["end_seconds"] if phrases else 0.0); end_index=min(len(ends)-1,cursor+max(0,length-1)); end=ends[end_index] if ends and end_index>=0 else start+max(1,length/5.2)
            phrases.append({"scene_id":segment["scene_id"],"start_seconds":round(float(start),3),"end_seconds":round(float(end),3),"display_text":segment["display_text"],"spoken_text":segment["spoken_text"]}); cursor+=length+1
        return {"provider":"elevenlabs","phrases":phrases,"duration_seconds":phrases[-1]["end_seconds"] if phrases else 0,"character_alignment":raw}

    def _analysis(self,facts:dict[str,Any])->dict[str,Any]:
        """Create fact-ID-only insights without permitting invented values."""
        return {"summary":"收入与数据中心业务创纪录，增长强劲；下一季指引进一步抬升。","insights":[{"claim":"收入同比增长 85%","fact_ids":["fact.revenue.q1fy27"]},{"claim":"数据中心占收入约 92%","fact_ids":["fact.datacenter.q1fy27","fact.revenue.q1fy27"],"derived_formula":"75.2/81.615"},{"claim":"Q2 收入指引中点 910 亿美元","fact_ids":["fact.guidance.q2fy27"]}]}

    def _story(self)->dict[str,Any]:
        """Build a compact narrative arc for retail investors."""
        target=self.request.content_requirements.target_duration_seconds if self.request.content_requirements else 70
        return {"hook":"一张成绩单，三个数字：AI 热潮到底有多强？","arc":["收入加速","数据中心主引擎","利润与下一季指引","风险与免责声明"],"target_duration_seconds":target,"fact_policy":"Every numeric claim must cite canonical fact_ids."}

    def _scenes(self)->dict[str,Any]:
        """Define scene-level retry boundaries and reusable subject references."""
        target=self.request.content_requirements.target_duration_seconds if self.request.content_requirements else 70; factor=target/70
        base=[{"id":"s1","purpose":"hook","duration_seconds":8,"asset_subject":"mascot-chip","transition":"zoom"},{"id":"s2","purpose":"revenue trend","duration_seconds":18,"chart":"revenue-bars","transition":"slide"},{"id":"s3","purpose":"data center mix","duration_seconds":16,"chart":"mix-donut","asset_subject":"mascot-chip","transition":"wipe"},{"id":"s4","purpose":"margin and guidance","duration_seconds":18,"chart":"guidance-range","transition":"fade"},{"id":"s5","purpose":"disclaimer","duration_seconds":10,"transition":"cut"}]
        for scene in base: scene["duration_seconds"]=round(max(4 if scene["purpose"]=="disclaimer" else 2,scene["duration_seconds"]*factor),2)
        return {"scenes":base}

    def _narration(self,scenes:dict[str,Any])->dict[str,Any]:
        """Generate separate display and normalized spoken text for each scene."""
        texts=["英伟达最新成绩单来了：AI 热潮，究竟有多强？","Q1 FY2027 收入 $81.6B，同比 +85%，环比 +20%，增长曲线再次变陡。","Data Center 收入 $75.2B，同比 +92%，约占总收入九成二，是最核心的增长引擎。","GAAP 毛利率 74.9%。公司给出的 Q2 FY2027 收入指引是 $91.0B，±2%。","但高增长不等于低风险。本视频仅供信息与产品演示用途，不构成投资建议。"]
        return {"language":"zh-CN","source":"generated","segments":[{"scene_id":s["id"],"display_text":t,"spoken_text":normalize_spoken(t),"fact_ids":ids} for s,t,ids in zip(scenes["scenes"],texts,[[],["fact.revenue.q1fy27"],["fact.datacenter.q1fy27","fact.revenue.q1fy27"],["fact.grossmargin.q1fy27","fact.guidance.q2fy27"],[]])]}

    def _resolve_narration(self,scenes:dict[str,Any],bundle:dict[str,Any]|None)->dict[str,Any]:
        """Honor transcript mode instead of always replacing a supplied script."""
        transcript=self.request.transcript
        if transcript.mode == "generate":
            narration=bundle["narration"] if bundle else self._narration(scenes)
            narration.setdefault("source","generated")
            return narration
        if transcript.allow_editing and bundle:
            narration=bundle["narration"]
            narration["source"]="pre-written"
            narration["editing_applied"]=True
            return narration
        return self._pre_written_narration(scenes)

    def _pre_written_narration(self,scenes:dict[str,Any])->dict[str,Any]:
        """Segment a locked user script without rewriting it and add the required disclaimer scene."""
        text=self.request.transcript.text or ""
        sentences=[part for part in re.split(r"(?<=[。！？!?])",text) if part]
        scene_list=scenes["scenes"]
        content_scenes=[scene for scene in scene_list if scene.get("purpose")!="disclaimer"]
        segments=[]
        for index,scene in enumerate(content_scenes):
            start=index*len(sentences)//max(1,len(content_scenes)); end=(index+1)*len(sentences)//max(1,len(content_scenes))
            display="".join(sentences[start:end])
            segments.append({"scene_id":scene["id"],"display_text":display,"spoken_text":normalize_spoken(display),"fact_ids":[]})
        for scene in scene_list:
            if scene.get("purpose")=="disclaimer":
                display=self.request.source_materials.disclaimer.strip()
                segments.append({"scene_id":scene["id"],"display_text":display,"spoken_text":normalize_spoken(display),"fact_ids":[]})
        return {"language":self.request.transcript.language,"source":"pre-written","editing_applied":False,"segments":segments}

    def _charts(self)->dict[str,Any]:
        """Declare chart data, animations, labels and subtitle-safe regions."""
        safe={"x":96,"y":756,"width":1728,"height":216}
        return {"reserved_regions":{"captions":safe},"charts":[{"id":"revenue-bars","type":"bar","labels":["Q1 FY26","Q4 FY26","Q1 FY27"],"values":[44.062,68.127,81.615],"unit":"USD billions","fact_ids":["fact.revenue.q1fy26","fact.revenue.q4fy26","fact.revenue.q1fy27"],"animation":"grow"},{"id":"mix-donut","type":"donut","values":[75.2,6.415],"labels":["Data Center","其他"],"fact_ids":["fact.datacenter.q1fy27","fact.revenue.q1fy27"],"animation":"sweep"},{"id":"guidance-range","type":"range","midpoint":91,"low":89.18,"high":92.82,"unit":"USD billions","fact_ids":["fact.guidance.q2fy27"],"key_levels":[{"kind":"guidance_low","value":89.18},{"kind":"guidance_mid","value":91},{"kind":"guidance_high","value":92.82}],"animation":"pan-and-highlight"}]}

    def _tts_alignment(self,narration:dict[str,Any])->dict[str,Any]:
        """Produce deterministic phrase timestamps; a real adapter replaces these with ElevenLabs timestamps."""
        cursor=0.0; phrases=[]
        for seg in narration["segments"]:
            duration=max(4.0,len(seg["spoken_text"])/5.2); phrases.append({"scene_id":seg["scene_id"],"start_seconds":round(cursor,3),"end_seconds":round(cursor+duration,3),"display_text":seg["display_text"],"spoken_text":seg["spoken_text"]}); cursor+=duration
        return {"provider":"deterministic-timestamp-adapter","phrases":phrases,"duration_seconds":round(cursor,3)}

    def _tts_audio(self,narration:dict[str,Any])->dict[str,Any]:
        """Render a zero-key Mandarin voice track with macOS say, or document the production adapter requirement."""
        spoken="。".join(segment["spoken_text"] for segment in narration["segments"]); say=shutil.which("say")
        public=ROOT/"public"; public.mkdir(parents=True,exist_ok=True); filename=f"{self.job_id}-narration.wav"; target=public/filename
        if say:
            aiff=self.run_dir/"narration-source.aiff"; subprocess.run([say,"-v","Tingting","-r","185","-o",str(aiff),spoken],check=True)
            afconvert=shutil.which("afconvert")
            if not afconvert: raise RuntimeError("macOS afconvert is required for the local TTS fallback")
            subprocess.run([afconvert,str(aiff),str(target),"-f","WAVE","-d","LEI16"],check=True)
            shutil.copy2(target,self.run_dir/"narration.wav")
            return {"provider":"macos-say-fallback","voice":"Tingting","characters":len(spoken),"artifact":"narration.wav","remotion_static_src":filename}
        raise RuntimeError("Deterministic narration requires macOS say and afconvert; use production ElevenLabs on other platforms")

    def _timeline(self,scenes:dict[str,Any],alignment:dict[str,Any])->dict[str,Any]:
        """Compile allowed transitions and animation events to frame boundaries."""
        fps=self.request.output.fps; events=[]; frame=0
        for scene in scenes["scenes"]:
            selected=scene.get("transition","cut"); transition=selected if selected in TRANSITIONS else "cut"; phrase=next((p for p in alignment.get("phrases",[]) if p["scene_id"]==scene["id"]),None); phrase_duration=(phrase["end_seconds"]-phrase["start_seconds"]+1.0) if phrase else 0; duration=max(float(scene["duration_seconds"]),phrase_duration); scene["duration_seconds"]=round(duration,3); frames=round(duration*fps); events.append({"scene_id":scene["id"],"from_frame":frame,"to_frame":frame+frames-1,"transition":{"template":transition,"duration_frames":min(round(.6*fps),frames//4)},"caption":phrase}); frame+=frames
        return {"fps":fps,"total_frames":frame,"allowed_transition_templates":sorted(TRANSITIONS),"events":events}

    def _assets(self,scenes:dict[str,Any])->dict[str,Any]:
        """Resolve one locked reference then apply the documented fallback ladder per scene."""
        refs={}; resolved=[]
        for scene in scenes["scenes"]:
            subject=scene.get("asset_subject")
            if subject and subject not in refs: refs[subject]={"reference_id":f"ref-{content_hash(subject)[:10]}","style_lock_id":"cartoon-v1","prompt_version":"1"}
            resolved.append({"scene_id":scene["id"],"strategy":"icon_illustration_gradient","attempts":[{"strategy":"bach_video","status":"skipped_no_key"},{"strategy":"bach_image_camera_motion","status":"skipped_no_key"},{"strategy":"icon_illustration_gradient","status":"succeeded"}],"reference_id":refs.get(subject,{}).get("reference_id")})
        return {"reference_registry":refs,"scenes":resolved}

    def _production_assets(self,scenes:dict[str,Any])->dict[str,Any]:
        """Route ordinary B-roll to T2V and repeated subjects to Elements, with two fallbacks."""
        adapter=BachAssets(self.settings); refs={}; resolved=[]; usage={"api_calls":0,"cost_usd":0,"retries":0}
        existing_assets=self.run_dir/"assets.json"
        if existing_assets.exists():
            refs.update(json.loads(existing_assets.read_text()).get("reference_registry",{}))
        subject_counts={}
        for scene in scenes["scenes"]:
            subject=scene.get("asset_subject") or scene.get("subject_id")
            if subject: subject_counts[subject]=subject_counts.get(subject,0)+1
        for subject,count in subject_counts.items():
            if count < 2 or refs.get(subject,{}).get("reference_input"): continue
            reference_id=f"ref-{content_hash(subject)[:10]}"; description=f"A friendly recurring {subject} financial explainer mascot, clean shapes, blue and green palette, suitable for retail investors"
            try:
                result,meta=adapter.text_to_subject(subject,description); usage["api_calls"]+=meta["api_calls"]; usage["cost_usd"]+=meta.get("cost_usd",0)
                images=list(result["split_images"].values())
                refs[subject]={"reference_id":reference_id,"style_lock_id":"production-cartoon-v1","prompt_version":"1","result":result,"reference_input":{"type":"subject","images":images,"subject":{"subject_name":subject[:50],"subject_desc":description[:500],"subject_style":"cartoon","subject_type":"character"}},"metrics":meta}
            except Exception as error:
                refs[subject]={"reference_id":reference_id,"style_lock_id":"production-cartoon-v1","prompt_version":"1","status":"fallback","reason":str(error)}
        for scene in scenes["scenes"]:
            subject=scene.get("asset_subject") or scene.get("subject_id")
            prompt=scene.get("visual_prompt") or f"Cartoon-style financial explainer B-roll: {scene.get('purpose','financial scene')}, clean blue and green palette, smooth camera motion"; attempts=[]
            if scene.get("purpose","").lower()=="disclaimer":
                resolved.append({"scene_id":scene["id"],"strategy":"icon_illustration_gradient","attempts":[{"strategy":"icon_illustration_gradient","status":"succeeded","reason":"disclaimer_uses_deterministic_template"}],"reference_id":None}); continue
            try:
                reference=refs.get(subject,{}).get("reference_input")
                if reference:
                    result,meta=adapter.elements_to_video(prompt,[reference]); strategy="bach_elements_to_video"
                else:
                    result,meta=adapter.text_to_video(prompt); strategy="bach_text_to_video"
                usage["api_calls"]+=meta["api_calls"]; usage["cost_usd"]+=meta.get("cost_usd",0); attempts.append({"strategy":strategy,"status":"succeeded","result":result,"metrics":meta})
            except Exception as video_error:
                attempted_strategy="bach_elements_to_video" if refs.get(subject,{}).get("reference_input") else "bach_text_to_video"; attempts.append({"strategy":attempted_strategy,"status":"failed","reason":str(video_error)}); usage["retries"]+=1
                try:
                    result,meta=adapter.text_to_image(prompt); usage["api_calls"]+=meta["api_calls"]; usage["cost_usd"]+=meta.get("cost_usd",0); attempts.append({"strategy":"bach_image_camera_motion","status":"succeeded","result":result,"metrics":meta}); strategy="bach_image_camera_motion"
                except Exception as image_error:
                    attempts.append({"strategy":"bach_image_camera_motion","status":"failed","reason":str(image_error)}); usage["retries"]+=1; attempts.append({"strategy":"icon_illustration_gradient","status":"succeeded"}); strategy="icon_illustration_gradient"
            resolved.append({"scene_id":scene["id"],"strategy":strategy,"attempts":attempts,"reference_id":refs.get(subject,{}).get("reference_id")})
        return {"reference_registry":refs,"scenes":resolved,"usage":usage}

    def _manifest(self,*parts:dict[str,Any])->dict[str,Any]:
        """Compile the renderer contract from all approved artifacts."""
        scenes,narration,charts,timeline,assets=parts
        tts=json.loads((self.run_dir/"tts.json").read_text())
        width,height=(int(v) for v in self.request.output.resolution.split("x")); caption_height=round(height*.2); margin=round(width*.05)
        return {"schema_version":"1.1","composition_id":"FinancialVideo","output":{"width":width,"height":height,"fps":self.request.output.fps,"codec":"h264"},"scenes":scenes["scenes"],"narration":narration,"audio":{"src":tts.get("remotion_static_src")},"charts":charts,"timeline":timeline,"assets":assets,"captions":{**self.request.captions.model_dump(mode="json"),"safe_zone":{"top":height-caption_height-round(height*.1),"bottom":height-round(height*.1),"left":margin,"right":width-margin}},"brand":self.request.brand.model_dump(mode="json"),"disclaimer":self.request.source_materials.disclaimer}

    def _render(self)->None:
        """Invoke Remotion; emit a clear placeholder only when dependencies are unavailable."""
        with self.logger.node("render",inputs=["render_manifest.json"]) as usage:
            remotion=ROOT/"node_modules/.bin/remotion"
            if shutil.which("node") and remotion.exists():
                started=time.time(); subprocess.run([str(remotion),"render","remotion/index.ts","FinancialVideo",str(self.run_dir/"final.mp4"),f"--props={self.run_dir/'render_manifest.json'}"],cwd=ROOT,check=True); usage["render_seconds"]=round(time.time()-started,2); usage["outputs"]=["final.mp4"]
            else: raise RuntimeError("Remotion dependencies or Node.js are unavailable; final MP4 was not rendered")
        self.record.artifacts+=usage["outputs"]

    def _qa(self,manifest:dict[str,Any])->None:
        """Run deterministic consistency, timeline, safe-zone and disclaimer checks."""
        width,height=manifest["output"]["width"],manifest["output"]["height"]; safe=manifest["captions"]["safe_zone"]
        audio_src=manifest.get("audio",{}).get("src"); audio_path=ROOT/"public"/audio_src if audio_src else None
        checks=[{"id":"facts_traceable","passed":all(isinstance(seg.get("fact_ids"),list) for seg in manifest["narration"]["segments"])},{"id":"transitions_template_bound","passed":all(e["transition"]["template"] in TRANSITIONS for e in manifest["timeline"]["events"])},{"id":"timeline_valid","passed":all(e["to_frame"]>=e["from_frame"] for e in manifest["timeline"]["events"])},{"id":"caption_safe_zone","passed":0<=safe["left"]<safe["right"]<=width and 0<=safe["top"]<safe["bottom"]<=height},{"id":"captions_nonempty","passed":not manifest["captions"]["enabled"] or all(bool(s["display_text"].strip()) for s in manifest["narration"]["segments"])},{"id":"audio_exists","passed":bool(audio_path and audio_path.exists() and audio_path.stat().st_size>0)},{"id":"disclaimer_present","passed":bool(manifest["disclaimer"].strip())},{"id":"disclaimer_duration","passed":manifest["scenes"][-1]["purpose"].lower()=="disclaimer" and manifest["scenes"][-1]["duration_seconds"]>=4},{"id":"video_exists","passed":(self.run_dir/"final.mp4").exists() and (self.run_dir/"final.mp4").stat().st_size>1024}]
        report={"passed":all(c["passed"] for c in checks),"checks":checks,"manual_review":{"required":True,"keyframes_seconds":[max(0,e["from_frame"]/manifest["output"]["fps"]+.5) for e in manifest["timeline"]["events"]]}}
        self._node("qa_report",report)
        if not report["passed"]: raise RuntimeError("QA failed: "+", ".join(c["id"] for c in checks if not c["passed"]))


def run_demo(job_id:str="nvidia-q1-fy27-demo",auto_approve:bool=True)->JobRecord:
    """Load the packaged NVIDIA input and execute a reproducible demo run."""
    request=JobInput.model_validate_json((ROOT/"examples/nvidia_q1_fy27/input.json").read_text())
    return Pipeline(job_id,request,auto_approve=auto_approve,settings=Settings.load("deterministic")).run()
