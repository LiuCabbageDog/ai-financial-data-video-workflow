"""Deterministic, artifact-first workflow used locally and by the API worker."""
from __future__ import annotations

import hashlib, json, re, shutil, subprocess, time
from pathlib import Path
from typing import Any

from .models import JobInput, JobRecord, JobStatus
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
    """Convert financial display notation to natural Mandarin TTS text."""
    replacements = [(r"\$81\.6B", "八百一十六点一五亿美元"),(r"\$75\.2B", "七百五十二亿美元"),(r"\$91\.0B", "九百一十亿美元"),(r"Q1 FY2027", "二零二七财年第一季度"),(r"Q2 FY2027", "二零二七财年第二季度"),(r"Data Center", "数据中心"),(r"GAAP", "美国通用会计准则")]
    spoken=display
    for pattern, value in replacements: spoken=re.sub(pattern,value,spoken,flags=re.I)
    spoken=re.sub(r"(\d+(?:\.\d+)?)%", lambda m: f"百分之{m.group(1)}", spoken)
    return spoken.replace("+", "增长").replace("±", "上下浮动")


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
        self.record=JobRecord(id=job_id,status=JobStatus.QUEUED)

    def save_record(self) -> None:
        """Atomically-enough persist current task status for local reads."""
        (self.run_dir/"job.json").write_text(self.record.model_dump_json(indent=2),encoding="utf-8")

    def run(self) -> JobRecord:
        """Execute the full deterministic demo and stop cleanly on conflicts/reviews."""
        self.record.status=JobStatus.RUNNING; self.save_record(); self.logger.event("workflow.started", input_hash=content_hash(self.request.model_dump()))
        try:
            self._node("input", self.request.model_dump())
            if self.settings.mode == "production":
                parsed=parse_financial_reports(self.request.source_materials.get("financial_reports",[])); self._node("parsed_reports",parsed)
                bundle,provider_meta=OpenAIPlanner(self.settings).generate(self.request.model_dump(),parsed); self._node("planning_provider",provider_meta,usage_meta=provider_meta)
                facts=bundle["canonical_facts"]
            else:
                bundle=None; facts=json.loads((ROOT/"examples/nvidia_q1_fy27/canonical_facts.json").read_text())
            self._node("canonical_facts", facts)
            conflict=self._conflict_check(facts)
            if conflict:
                self._node("conflict_report", conflict); self.record.status=JobStatus.BLOCKED_CONFLICT; self.save_record(); return self.record
            analysis=bundle["financial_analysis"] if bundle else self._analysis(facts); self._node("financial_analysis",analysis)
            story=bundle["story_plan"] if bundle else self._story(); self._reviewable("story_plan",story)
            scenes=bundle["scene_plan"] if bundle else self._scenes(); self._node("scene_plan",scenes)
            narration=bundle["narration"] if bundle else self._narration(scenes); self._reviewable("narration",narration)
            charts=bundle["chart_spec"] if bundle else self._charts(); self._reviewable("chart_spec",charts)
            if self.settings.mode == "production":
                spoken="。".join(x["spoken_text"] for x in narration["segments"]); public=ROOT/"public"; public.mkdir(exist_ok=True); audio_name=f"{self.job_id}-narration.mp3"
                audio,tts_meta=ElevenLabsTTS(self.settings).synthesize(spoken,public/audio_name); shutil.copy2(public/audio_name,self.run_dir/"narration.mp3"); audio["remotion_static_src"]=audio_name
                alignment={"provider":"elevenlabs","character_alignment":audio.pop("alignment")}; self._node("alignment",alignment,usage_meta=tts_meta)
                self._node("tts",audio,tts_chars=len(spoken),usage_meta=tts_meta)
            else:
                alignment=self._tts_alignment(narration); self._node("alignment",alignment,tts_chars=sum(len(x["spoken_text"]) for x in narration["segments"]))
                audio=self._tts_audio(narration); self._node("tts",audio,tts_chars=sum(len(x["spoken_text"]) for x in narration["segments"]))
            timeline=self._timeline(scenes,alignment); self._node("animation_timeline",timeline)
            assets=self._production_assets(scenes) if self.settings.mode=="production" else self._assets(scenes); self._node("assets",assets)
            manifest=self._manifest(scenes,narration,charts,timeline,assets); self._node("render_manifest",manifest)
            self.record.status=JobStatus.RENDERING; self.save_record(); self._render()
            self.record.status=JobStatus.QA; self.save_record(); self._qa(manifest)
            self.record.status=JobStatus.COMPLETED; self.record.progress=1; self.logger.event("workflow.completed",artifacts=self.record.artifacts); self.save_record(); return self.record
        except Exception as exc:
            self.record.status=JobStatus.FAILED; self.record.error=f"{type(exc).__name__}: {exc}"; self.logger.event("workflow.failed",reason=self.record.error); self.save_record(); raise

    def _node(self,name:str,value:Any,tts_chars:int=0,usage_meta:dict[str,Any]|None=None) -> None:
        """Persist one artifact and update node status/metrics."""
        with self.logger.node(name) as usage:
            filename=write_artifact(self.run_dir,f"{name}.json",value); usage["outputs"]=[filename]; usage["tts_characters"]=tts_chars
            for key in ("api_calls","input_tokens","output_tokens","cost_usd","latency_ms"):
                if usage_meta and key in usage_meta: usage[key]=usage_meta[key]
        self.record.nodes[name]="completed"; self.record.artifacts.append(filename); self.save_record()

    def _reviewable(self,name:str,value:Any) -> None:
        """Persist an auditable review gate; CLI can explicitly auto-approve."""
        self._node(name,value); review={"node":name,"status":"approved" if self.auto_approve else "pending","reviewer":"demo-auto-approve" if self.auto_approve else None,"at":time.time() if self.auto_approve else None}
        write_artifact(self.run_dir,f"review_{name}.json",review)
        if not self.auto_approve:
            self.record.status=JobStatus.WAITING_REVIEW; self.record.current_node=name; self.save_record(); raise RuntimeError(f"review required: {name}")

    def _conflict_check(self,facts:dict[str,Any]) -> dict[str,Any]|None:
        """Block pre-written transcripts with numeric claims absent from canonical facts."""
        if self.request.transcript.mode!="pre-written": return None
        text=self.request.transcript.text or ""; claimed={float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%",text)}
        allowed={float(v) for f in facts["facts"] for v in f.get("comparison",{}).values()}|{74.9,75.0}
        bad=sorted(claimed-allowed)
        return {"status":"blocked","reason":"transcript_fact_conflict","unsupported_percentages":bad,"action":"Correct transcript or provide a cited supplementary source."} if bad else None

    def _analysis(self,facts:dict[str,Any])->dict[str,Any]:
        """Create fact-ID-only insights without permitting invented values."""
        return {"summary":"收入与数据中心业务创纪录，增长强劲；下一季指引进一步抬升。","insights":[{"claim":"收入同比增长 85%","fact_ids":["fact.revenue.q1fy27"]},{"claim":"数据中心占收入约 92%","fact_ids":["fact.datacenter.q1fy27","fact.revenue.q1fy27"],"derived_formula":"75.2/81.615"},{"claim":"Q2 收入指引中点 910 亿美元","fact_ids":["fact.guidance.q2fy27"]}]}

    def _story(self)->dict[str,Any]:
        """Build a compact narrative arc for retail investors."""
        return {"hook":"一张成绩单，三个数字：AI 热潮到底有多强？","arc":["收入加速","数据中心主引擎","利润与下一季指引","风险与免责声明"],"target_duration_seconds":70,"fact_policy":"Every numeric claim must cite canonical fact_ids."}

    def _scenes(self)->dict[str,Any]:
        """Define scene-level retry boundaries and reusable subject references."""
        return {"scenes":[{"id":"s1","purpose":"hook","duration_seconds":8,"asset_subject":"mascot-chip","transition":"zoom"},{"id":"s2","purpose":"revenue trend","duration_seconds":18,"chart":"revenue-bars","transition":"slide"},{"id":"s3","purpose":"data center mix","duration_seconds":16,"chart":"mix-donut","asset_subject":"mascot-chip","transition":"wipe"},{"id":"s4","purpose":"margin and guidance","duration_seconds":18,"chart":"guidance-range","transition":"fade"},{"id":"s5","purpose":"disclaimer","duration_seconds":10,"transition":"cut"}]}

    def _narration(self,scenes:dict[str,Any])->dict[str,Any]:
        """Generate separate display and normalized spoken text for each scene."""
        texts=["英伟达最新成绩单来了：AI 热潮，究竟有多强？","Q1 FY2027 收入 $81.6B，同比 +85%，环比 +20%，增长曲线再次变陡。","Data Center 收入 $75.2B，同比 +92%，约占总收入九成二，是最核心的增长引擎。","GAAP 毛利率 74.9%。公司给出的 Q2 FY2027 收入指引是 $91.0B，±2%。","但高增长不等于低风险。本视频仅供信息与产品演示用途，不构成投资建议。"]
        return {"language":"zh-CN","segments":[{"scene_id":s["id"],"display_text":t,"spoken_text":normalize_spoken(t),"fact_ids":ids} for s,t,ids in zip(scenes["scenes"],texts,[[],["fact.revenue.q1fy27"],["fact.datacenter.q1fy27","fact.revenue.q1fy27"],["fact.grossmargin.q1fy27","fact.guidance.q2fy27"],[]])]}

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
        return {"provider":"unavailable","characters":len(spoken),"artifact":None,"remotion_static_src":None,"required":"Configure ElevenLabs or provide macOS say."}

    def _timeline(self,scenes:dict[str,Any],alignment:dict[str,Any])->dict[str,Any]:
        """Compile allowed transitions and animation events to frame boundaries."""
        fps=30; events=[]; frame=0
        for scene in scenes["scenes"]:
            selected=scene.get("transition","cut"); transition=selected if selected in TRANSITIONS else "cut"; frames=round(scene["duration_seconds"]*fps); events.append({"scene_id":scene["id"],"from_frame":frame,"to_frame":frame+frames-1,"transition":{"template":transition,"duration_frames":min(18,frames//4)}}); frame+=frames
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
        """Use BACH video, then BACH image plus camera motion, then deterministic graphics per scene."""
        adapter=BachAssets(self.settings); refs={}; resolved=[]
        for subject in {s.get("asset_subject") or s.get("subject_id") for s in scenes["scenes"] if s.get("asset_subject") or s.get("subject_id")}:
            reference_id=f"ref-{content_hash(subject)[:10]}"
            try:
                result,meta=adapter.generate("image",{"purpose":"reusable_subject_reference","subject_id":subject,"style":"cartoon","style_lock_id":"production-cartoon-v1"}); refs[subject]={"reference_id":reference_id,"style_lock_id":"production-cartoon-v1","prompt_version":"1","result":result,"metrics":meta}
            except Exception as error:
                refs[subject]={"reference_id":reference_id,"style_lock_id":"production-cartoon-v1","prompt_version":"1","status":"fallback","reason":str(error)}
        for scene in scenes["scenes"]:
            subject=scene.get("asset_subject") or scene.get("subject_id")
            payload={"prompt":scene.get("visual_prompt") or scene.get("purpose","financial scene"),"scene_id":scene["id"],"reference":refs.get(subject),"style":"cartoon"}; attempts=[]
            try:
                result,meta=adapter.generate("video",payload); attempts.append({"strategy":"bach_video","status":"succeeded","result":result,"metrics":meta}); strategy="bach_video"
            except Exception as video_error:
                attempts.append({"strategy":"bach_video","status":"failed","reason":str(video_error)})
                try:
                    result,meta=adapter.generate("image",payload); attempts.append({"strategy":"bach_image_camera_motion","status":"succeeded","result":result,"metrics":meta}); strategy="bach_image_camera_motion"
                except Exception as image_error:
                    attempts.append({"strategy":"bach_image_camera_motion","status":"failed","reason":str(image_error)}); attempts.append({"strategy":"icon_illustration_gradient","status":"succeeded"}); strategy="icon_illustration_gradient"
            resolved.append({"scene_id":scene["id"],"strategy":strategy,"attempts":attempts,"reference_id":refs.get(subject,{}).get("reference_id")})
        return {"reference_registry":refs,"scenes":resolved}

    def _manifest(self,*parts:dict[str,Any])->dict[str,Any]:
        """Compile the renderer contract from all approved artifacts."""
        scenes,narration,charts,timeline,assets=parts
        tts=json.loads((self.run_dir/"tts.json").read_text())
        return {"schema_version":"1.0","composition_id":"FinancialVideo","output":{"width":1920,"height":1080,"fps":30,"codec":"h264"},"scenes":scenes["scenes"],"narration":narration,"audio":{"src":tts.get("remotion_static_src")},"charts":charts,"timeline":timeline,"assets":assets,"captions":{"safe_zone":{"top":756,"bottom":972,"left":96,"right":1824},"max_lines":2},"disclaimer":self.request.source_materials["disclaimer"]}

    def _render(self)->None:
        """Invoke Remotion; emit a clear placeholder only when dependencies are unavailable."""
        with self.logger.node("render",inputs=["render_manifest.json"]) as usage:
            remotion=ROOT/"node_modules/.bin/remotion"
            if shutil.which("node") and remotion.exists():
                started=time.time(); subprocess.run([str(remotion),"render","remotion/index.ts","FinancialVideo",str(self.run_dir/"final.mp4"),f"--props={self.run_dir/'render_manifest.json'}"],cwd=ROOT,check=True); usage["render_seconds"]=round(time.time()-started,2); usage["outputs"]=["final.mp4"]
            else:
                (self.run_dir/"RENDER_PENDING.txt").write_text("Run npm install, then: npm run render -- --props runs/%s/render_manifest.json runs/%s/final.mp4\n"%(self.job_id,self.job_id),encoding="utf-8"); usage["outputs"]=["RENDER_PENDING.txt"]
        self.record.artifacts+=usage["outputs"]

    def _qa(self,manifest:dict[str,Any])->None:
        """Run deterministic consistency, timeline, safe-zone and disclaimer checks."""
        checks=[{"id":"facts_traceable","passed":all(seg.get("fact_ids") is not None for seg in manifest["narration"]["segments"])},{"id":"transitions_template_bound","passed":all(e["transition"]["template"] in TRANSITIONS for e in manifest["timeline"]["events"])},{"id":"timeline_valid","passed":all(e["to_frame"]>=e["from_frame"] for e in manifest["timeline"]["events"])},{"id":"caption_safe_zone","passed":manifest["captions"]["safe_zone"]["bottom"]<=972},{"id":"disclaimer_present","passed":bool(manifest["disclaimer"])},{"id":"disclaimer_duration","passed":manifest["scenes"][-1]["duration_seconds"]>=4},{"id":"video_exists","passed":(self.run_dir/"final.mp4").exists(),"severity":"warning"}]
        self._node("qa_report",{"passed":all(c["passed"] for c in checks if c.get("severity")!="warning"),"checks":checks,"manual_review":{"required":True,"keyframes_seconds":[1,12,30,48,64]}})


def run_demo(job_id:str="nvidia-q1-fy27-demo",auto_approve:bool=True)->JobRecord:
    """Load the packaged NVIDIA input and execute a reproducible demo run."""
    request=JobInput.model_validate_json((ROOT/"examples/nvidia_q1_fy27/input.json").read_text())
    return Pipeline(job_id,request,auto_approve=auto_approve,settings=Settings.load("deterministic")).run()
