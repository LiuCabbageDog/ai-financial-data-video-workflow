"""Deterministic, artifact-first workflow used locally and by the API worker."""
from __future__ import annotations

import hashlib, json, math, re, shutil, subprocess, time
from decimal import Decimal
from pathlib import Path
from typing import Any

from .models import CanonicalFacts, ChartSpec, JobInput, JobRecord, JobStatus, Narration, ScenePlan
from .observability import RunLogger
from .config import Settings
from .adapters import BachAssets, ElevenLabsTTS, OpenAIPlanner, parse_financial_reports

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REVIEW_NODES = {"story_plan", "narration", "chart_spec"}


def money_base_units(fact:dict[str,Any]) -> float:
    """Convert a currency fact's declared scale to base monetary units."""
    scale=str(fact.get("scale","ones")).strip().lower().replace("_","-")
    multipliers={
        "one":1.0,"ones":1.0,"unit":1.0,"units":1.0,"per-share":1.0,
        "thousand":1e3,"thousands":1e3,"k":1e3,
        "million":1e6,"millions":1e6,"m":1e6,
        "billion":1e9,"billions":1e9,"b":1e9,
        "trillion":1e12,"trillions":1e12,"t":1e12,
    }
    multiplier=multipliers.get(scale)
    if multiplier is None: raise ValueError(f"unsupported currency scale: {fact.get('scale')}")
    return float(fact["value"])*multiplier


def percentage_claim_supported(claim:str,allowed:set[float]) -> bool:
    """Match a displayed percentage using its stated decimal precision."""
    decimals=len(claim.partition(".")[2]) if "." in claim else 0
    rounding_tolerance=.5*(10**-decimals)
    value=float(claim)
    return any(abs(value-candidate)<=rounding_tolerance+1e-9 for candidate in allowed)


def is_disclaimer_scene(scene:dict[str,Any]) -> bool:
    """Identify disclaimer scenes by stable kind, with legacy-language fallback."""
    if scene.get("kind") in {"content","disclaimer"}: return scene["kind"]=="disclaimer"
    text=f"{scene.get('purpose','')} {scene.get('title','')}".lower()
    return "disclaimer" in text or "免责声明" in text or "法律提示" in text
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
    multipliers={"K":Decimal("1000"),"M":Decimal("1000000"),"B":Decimal("1000000000"),"T":Decimal("1000000000000")}
    def money(match:re.Match[str])->str:
        amount=Decimal(match.group(2))*multipliers.get((match.group(3) or "").upper(),Decimal("1")); currency="美元" if match.group(1)=="$" else "元"
        per_share=bool(match.group(4)); rendered=_spoken_money(amount)
        return f"每股{rendered}{currency}" if per_share else f"{rendered}{currency}"
    spoken=re.sub(r"([$¥€£])\s*(-?\d+(?:\.\d+)?)\s*([KMBT])?\s*(/股|每股)?",money,spoken,flags=re.I)
    spoken=re.sub(r"(-?\d+(?:\.\d+)?)%",lambda m:f"百分之{_spoken_number(Decimal(m.group(1)))}",spoken)
    return spoken.replace("+", "增长").replace("±", "上下浮动").replace("−","负").replace("-", "负")


def _spoken_integer(value:int)->str:
    """Render a non-negative integer with standard Mandarin section units."""
    if value==0: return "零"
    digits="零一二三四五六七八九"; section_units=["","万","亿","万亿"]
    sections=[]
    while value: sections.append(value%10000); value//=10000
    out=""; pending_zero=False
    for index in range(len(sections)-1,-1,-1):
        section=sections[index]
        if not section: pending_zero=bool(out); continue
        if out and (pending_zero or section<1000): out+="零"
        chars=""; zero=False
        for divisor,unit in ((1000,"千"),(100,"百"),(10,"十"),(1,"")):
            digit=section//divisor; section%=divisor
            if digit:
                if zero and chars: chars+="零"
                chars+=digits[digit]+unit; zero=False
            elif chars and section: zero=True
        out+=chars+section_units[index]; pending_zero=False
    return out[1:] if out.startswith("一十") else out


def _spoken_number(value:Decimal)->str:
    sign="负" if value<0 else ""; rendered=format(abs(value),"f")
    if "." in rendered: rendered=rendered.rstrip("0").rstrip(".")
    whole,dot,fraction=rendered.partition("."); result=_spoken_integer(int(whole))
    if dot: result+="点"+"".join("零一二三四五六七八九"[int(d)] for d in fraction)
    return sign+result


def _spoken_money(amount:Decimal)->str:
    absolute=abs(amount); sign="负" if amount<0 else ""
    for divisor,unit in ((Decimal("100000000"),"亿"),(Decimal("10000"),"万")):
        if absolute>=divisor and absolute%divisor==0: return sign+_spoken_number(absolute/divisor)+unit
    return _spoken_number(amount)


def visual_summary(scenes:dict[str,Any],narration:dict[str,Any]|None=None)->dict[str,Any]:
    """Return the common review, manifest, and QA data-visual metrics."""
    content=[scene for scene in scenes["scenes"] if not is_disclaimer_scene(scene)]
    count=max(1,len(content)); data=sum(scene.get("visual_kind") in {"chart","metric_cards"} for scene in content)
    return {"data_visual_coverage":round(data/count,3),"chart_count":sum(scene.get("visual_kind")=="chart" for scene in content),"metric_card_count":sum(scene.get("visual_kind")=="metric_cards" for scene in content),"broll_count":sum(scene.get("visual_kind")=="broll" for scene in content),"content_scene_count":len(content),"estimated_audio_seconds":round(sum(len(s["spoken_text"]) for s in (narration or {}).get("segments",[]))/4.8,2)}


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
        allowed={"purpose","title","duration_seconds","visual_kind","visual_prompt","transition","chart","asset_subject","subject_id"}
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
            scenes=ScenePlan.model_validate(bundle["scene_plan"] if bundle else self._scenes()).model_dump(mode="json"); self._validate_visual_mix(scenes)
            narration=self._validate_narration(self._resolve_narration(scenes,bundle),scenes,facts)
            self._node("review_summary",visual_summary(scenes,narration))
            if not self._reviewable("story_plan",story): return self.record
            self._node("scene_plan",scenes)
            if not self._reviewable("narration",narration): return self.record
            charts=ChartSpec.model_validate(bundle["chart_spec"] if bundle else self._charts()).model_dump(mode="json"); self._validate_charts(charts,scenes,facts)
            if not self._reviewable("chart_spec",charts): return self.record
            if self.settings.mode == "production":
                spoken,spans=self._tts_script(narration); public=ROOT/"public"; public.mkdir(exist_ok=True); audio_name=f"{self.job_id}-narration.mp3"
                adapter=ElevenLabsTTS(self.settings); audio,tts_meta=adapter.synthesize(spoken,public/audio_name,self.settings.elevenlabs_speed)
                alignment=self._phrase_alignment(spans,audio.pop("alignment")); initial_duration=alignment["duration_seconds"]
                adjusted_speed=self._duration_adjusted_speed(initial_duration,self.settings.elevenlabs_speed)
                if adjusted_speed is not None:
                    audio,fit_meta=adapter.synthesize(spoken,public/audio_name,adjusted_speed); alignment=self._phrase_alignment(spans,audio.pop("alignment"))
                    audio["duration_fit"]={"initial_duration_seconds":initial_duration,"initial_speed":self.settings.elevenlabs_speed,"adjusted_speed":adjusted_speed,"adjusted_duration_seconds":alignment["duration_seconds"]}
                    for key in ("api_calls","cost_usd","latency_ms"): tts_meta[key]=tts_meta.get(key,0)+fit_meta.get(key,0)
                    tts_meta["voice_settings"]=fit_meta["voice_settings"]
                shutil.copy2(public/audio_name,self.run_dir/"narration.mp3"); audio["remotion_static_src"]=audio_name; self._node("alignment",alignment)
                self._validate_target_duration(alignment,audio.get("duration_fit"))
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
        allowed_pct={round(float(v),4) for f in fact_list for v in f.get("comparison",{}).values()}|{round(float(f["value"]),4) for f in fact_list if str(f.get("unit","")).lower() in {"percent","%"}}
        claimed_pct_tokens=re.findall(r"(-?\d+(?:\.\d+)?)\s*%",text); claimed_pct={round(float(x),4) for x in claimed_pct_tokens}
        scales={"K":1e3,"M":1e6,"B":1e9,"T":1e12}; claimed_money=[]
        for amount,scale in re.findall(r"[$¥€£]\s*(-?\d+(?:\.\d+)?)\s*([KMBT])\b",text,re.I): claimed_money.append(float(amount)*scales[scale.upper()])
        allowed_money=[money_base_units(f) for f in fact_list if f.get("currency")]
        bad_money=[v for v in claimed_money if not any(abs(v-a)<=max(1,abs(a)*.006) for a in allowed_money)]
        periods=set(re.findall(r"Q[1-4]\s*FY\d{4}",text,re.I)); allowed_periods={str(f.get("fiscal_period","")).replace(" ","").upper() for f in fact_list}
        bad_periods=sorted(p for p in periods if p.replace(" ","").upper() not in allowed_periods)
        direction_error=bool(re.search(r"(?:下降|减少|down|declin)",text,re.I) and any(v>0 for v in claimed_pct) and claimed_pct & {abs(v) for v in allowed_pct if v>0})
        unsupported_pct=sorted({float(value) for value in claimed_pct_tokens if not percentage_claim_supported(value,allowed_pct)})
        report={"status":"blocked","reason":"transcript_fact_conflict","unsupported_percentages":unsupported_pct,"unsupported_money_base_units":bad_money,"unsupported_periods":bad_periods,"direction_conflict":direction_error,"action":"Correct transcript or provide a cited supplementary source."}
        return report if any([report["unsupported_percentages"],bad_money,bad_periods,direction_error]) else None

    def _validate_narration(self,narration:dict[str,Any],scenes:dict[str,Any],facts:dict[str,Any])->dict[str,Any]:
        """Normalize spoken text and enforce references after fact-part compilation."""
        narration=Narration.model_validate(narration).model_dump(mode="json"); scene_ids={s["id"] for s in scenes["scenes"]}; fact_ids={f["id"] for f in facts["facts"]}
        for segment in narration["segments"]:
            if segment["scene_id"] not in scene_ids: raise ValueError(f"narration references unknown scene: {segment['scene_id']}")
            missing=set(segment["fact_ids"])-fact_ids
            if missing: raise ValueError(f"narration references unknown facts: {sorted(missing)}")
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

    def _validate_target_duration(self,alignment:dict[str,Any],duration_fit:dict[str,Any]|None=None) -> None:
        """Stop before asset generation when generated speech misses the requested duration."""
        if not self.request.content_requirements: return
        target=float(self.request.content_requirements.target_duration_seconds); actual=float(alignment.get("duration_seconds",0)); tolerance=self.settings.target_duration_tolerance
        if not target*(1-tolerance)<=actual<=target*(1+tolerance):
            details=""
            if duration_fit: details=f"; initial={float(duration_fit['initial_duration_seconds']):.1f}s, adjusted_speed={float(duration_fit['adjusted_speed']):.3f}, second={actual:.1f}s"
            raise RuntimeError(f"Actual narration duration {actual:.1f}s is outside target {target:.1f}s ±{tolerance:.0%} after duration fitting{details}")

    def _duration_adjusted_speed(self,actual_duration:float,base_speed:float) -> float|None:
        """Return one bounded TTS speed correction when actual generated audio misses target."""
        if not self.settings.tts_auto_fit_duration or not self.request.content_requirements: return None
        target=float(self.request.content_requirements.target_duration_seconds); tolerance=self.settings.target_duration_tolerance
        if target*(1-tolerance)<=actual_duration<=target*(1+tolerance): return None
        adjusted=round(max(.9,min(1.2,base_speed*actual_duration/target)),3)
        return adjusted if abs(adjusted-base_speed)>=.005 else None

    def _validate_visual_mix(self,scenes:dict[str,Any])->None:
        """Enforce the data-first scene contract in every adapter mode."""
        summary=visual_summary(scenes); content=summary["content_scene_count"]
        if summary["data_visual_coverage"]<.8: raise ValueError("data visual coverage must be at least 80%")
        if summary["chart_count"]<2: raise ValueError("at least two chart scenes are required")
        if summary["metric_card_count"]<1: raise ValueError("at least one metric_cards scene is required")
        if summary["broll_count"]>math.floor(content*.2): raise ValueError("broll may occupy at most 20% of content scenes")

    def _tts_script(self,narration:dict[str,Any])->tuple[str,list[dict[str,Any]]]:
        """Build the exact TTS string and character spans for short caption clauses."""
        script=""; spans=[]
        for segment in narration["segments"]:
            if script: script+="\n"
            # Do not split on an ASCII period: it is commonly a decimal point
            # in captions such as $81.6B and caused whole-scene cue fallbacks.
            spoken_parts=[p for p in re.split(r"(?<=[，。！？；：,!?;:])",segment["spoken_text"]) if p]
            display_parts=[p for p in re.split(r"(?<=[，。！？；：,!?;:])",segment["display_text"]) if p]
            if len(display_parts)!=len(spoken_parts): display_parts=[segment["display_text"]]*len(spoken_parts)
            for index,part in enumerate(spoken_parts):
                start=len(script); script+=part; spans.append({"scene_id":segment["scene_id"],"text":display_parts[index].strip(),"spoken_text":part.strip(),"start_char":start,"end_char":len(script)-1})
        return script,spans

    def _phrase_alignment(self,spans:list[dict[str,Any]],raw:dict[str,Any])->dict[str,Any]:
        """Compile precise short-clause cues from ElevenLabs character timestamps."""
        if isinstance(spans,dict):
            _,spans=self._tts_script(spans)
        starts=raw.get("character_start_times_seconds",[]); ends=raw.get("character_end_times_seconds",[]); cues=[]
        script_length=max((span["end_char"] for span in spans),default=-1)+1
        aligned_length=min(len(starts),len(ends)); exact_indices=aligned_length==script_length
        def aligned_index(character_index:int,is_end:bool=False)->int:
            if not aligned_length: return 0
            if exact_indices: return min(character_index,aligned_length-1)
            # Safe fallback for normalized alignments (for example Mandarin
            # expanded to pinyin): preserve the global temporal proportion.
            ratio=(character_index+1 if is_end else character_index)/max(1,script_length)
            projected=math.ceil(ratio*aligned_length)-1 if is_end else math.floor(ratio*aligned_length)
            return min(aligned_length-1,max(0,projected))
        for span in spans:
            start_i=aligned_index(span["start_char"]); end_i=aligned_index(span["end_char"],True)
            start=float(starts[start_i]) if starts else (cues[-1]["end_seconds"] if cues else 0)
            end=float(ends[end_i]) if ends else start+max(.3,len(span["spoken_text"])/4.2)
            cues.append({**{k:span[k] for k in ("scene_id","text","spoken_text")},"start_seconds":round(start,3),"end_seconds":round(max(start,end),3)})
        phrases=[]
        for scene_id in dict.fromkeys(c["scene_id"] for c in cues):
            group=[c for c in cues if c["scene_id"]==scene_id]; phrases.append({"scene_id":scene_id,"start_seconds":group[0]["start_seconds"],"end_seconds":group[-1]["end_seconds"],"display_text":"".join(c["text"] for c in group),"spoken_text":"".join(c["spoken_text"] for c in group)})
        duration=float(ends[-1]) if ends else (cues[-1]["end_seconds"] if cues else 0)
        return {"provider":"elevenlabs","phrases":phrases,"caption_cues":cues,"duration_seconds":round(duration,3),"alignment_index_mode":"exact" if exact_indices else "proportional_fallback","character_alignment":raw}

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
        base=[{"id":"s1","kind":"content","visual_kind":"broll","purpose":"hook","duration_seconds":7,"asset_subject":"mascot-chip","transition":"zoom"},{"id":"s2","kind":"content","visual_kind":"chart","purpose":"revenue trend","duration_seconds":15,"chart":"revenue-bars","transition":"slide"},{"id":"s3","kind":"content","visual_kind":"chart","purpose":"data center mix","duration_seconds":14,"chart":"mix-donut","transition":"wipe"},{"id":"s4","kind":"content","visual_kind":"chart","purpose":"margin trend","duration_seconds":13,"chart":"margin-line","transition":"fade"},{"id":"s5","kind":"content","visual_kind":"metric_cards","purpose":"guidance and outlook","duration_seconds":13,"chart":"guidance-cards","transition":"slide"},{"id":"s6","kind":"disclaimer","visual_kind":"disclaimer","purpose":"disclaimer","duration_seconds":8,"transition":"cut"}]
        for scene in base: scene["duration_seconds"]=round(max(4 if is_disclaimer_scene(scene) else 2,scene["duration_seconds"]*factor),2)
        return {"scenes":base}

    def _narration(self,scenes:dict[str,Any])->dict[str,Any]:
        """Generate separate display and normalized spoken text for each scene."""
        texts=["英伟达最新成绩单来了：这轮人工智能需求，究竟如何反映在财务数据里？","Q1 FY2027 收入 $81.6B，同比 +85%，环比 +20%。连续增长说明算力投资仍在加速兑现。","Data Center 收入 $75.2B，同比 +92%，已经成为最核心的增长引擎，也决定了整体收入弹性。","GAAP 毛利率 74.9%。在收入高速扩张的同时，利润率仍保持高位，是本季质量的重要观察点。","公司给出的 Q2 FY2027 收入指引是 $91.0B，±2%。接下来需要同时关注需求兑现、供应能力和利润率变化。","高增长并不等于低风险。本视频仅供信息与产品演示用途，不构成投资建议。"]
        refs=[[],["fact.revenue.q1fy27"],["fact.datacenter.q1fy27","fact.revenue.q1fy27"],["fact.grossmargin.q1fy27"],["fact.guidance.q2fy27"],[]]
        return {"language":"zh-CN","source":"generated","segments":[{"scene_id":s["id"],"display_text":t,"spoken_text":normalize_spoken(t),"fact_ids":ids} for s,t,ids in zip(scenes["scenes"],texts,refs)]}

    def _resolve_narration(self,scenes:dict[str,Any],bundle:dict[str,Any]|None)->dict[str,Any]:
        """Honor transcript mode and force the request disclaimer into narration."""
        transcript=self.request.transcript
        if transcript.mode == "generate":
            narration=bundle["narration"] if bundle else self._narration(scenes)
            narration.setdefault("source","generated")
        elif transcript.allow_editing and bundle:
            narration=bundle["narration"]
            narration["source"]="pre-written"
            narration["editing_applied"]=True
        else:
            narration=self._pre_written_narration(scenes)
        disclaimer_ids={scene["id"] for scene in scenes["scenes"] if is_disclaimer_scene(scene)}
        disclaimer=self.request.source_materials.disclaimer.strip()
        replacement={"display_text":disclaimer,"spoken_text":normalize_spoken(disclaimer),"fact_ids":[]}
        segments=[]; replaced=set()
        for segment in narration["segments"]:
            if segment["scene_id"] in disclaimer_ids:
                segments.append({**segment,**replacement}); replaced.add(segment["scene_id"])
            else:
                segments.append(dict(segment))
        for scene in scenes["scenes"]:
            if scene["id"] in disclaimer_ids and scene["id"] not in replaced:
                segments.append({"scene_id":scene["id"],**replacement})
        return {**narration,"segments":segments}

    def _pre_written_narration(self,scenes:dict[str,Any])->dict[str,Any]:
        """Segment a locked user script without rewriting it and add the required disclaimer scene."""
        text=self.request.transcript.text or ""
        sentences=[part for part in re.split(r"(?<=[。！？!?])",text) if part]
        scene_list=scenes["scenes"]
        content_scenes=[scene for scene in scene_list if not is_disclaimer_scene(scene)]
        segments=[]
        for index,scene in enumerate(content_scenes):
            start=index*len(sentences)//max(1,len(content_scenes)); end=(index+1)*len(sentences)//max(1,len(content_scenes))
            display="".join(sentences[start:end])
            segments.append({"scene_id":scene["id"],"display_text":display,"spoken_text":normalize_spoken(display),"fact_ids":[]})
        for scene in scene_list:
            if is_disclaimer_scene(scene):
                display=self.request.source_materials.disclaimer.strip()
                segments.append({"scene_id":scene["id"],"display_text":display,"spoken_text":normalize_spoken(display),"fact_ids":[]})
        return {"language":self.request.transcript.language,"source":"pre-written","editing_applied":False,"segments":segments}

    def _charts(self)->dict[str,Any]:
        """Declare chart data, animations, labels and subtitle-safe regions."""
        safe={"x":96,"y":756,"width":1728,"height":216}
        return {"reserved_regions":{"captions":safe},"charts":[{"id":"revenue-bars","type":"bar","labels":["Q1 FY26","Q4 FY26","Q1 FY27"],"values":[44.062,68.127,81.615],"unit":"USD billions","fact_ids":["fact.revenue.q1fy26","fact.revenue.q4fy26","fact.revenue.q1fy27"],"animation":"grow"},{"id":"mix-donut","type":"donut","values":[75.2,6.415],"labels":["Data Center","其他"],"fact_ids":["fact.datacenter.q1fy27","fact.revenue.q1fy27"],"animation":"sweep"},{"id":"margin-line","type":"line","title":"GAAP 毛利率","labels":["Q1 FY27"],"values":[74.9],"unit":"percent","fact_ids":["fact.grossmargin.q1fy27"],"animation":"draw-and-highlight"},{"id":"guidance-cards","type":"metric_cards","title":"下一季度指引","labels":["中点","低值","高值"],"values":[91,89.18,92.82],"unit":"USD billions","fact_ids":["fact.guidance.q2fy27"],"key_levels":[],"animation":"grow"}]}

    def _tts_alignment(self,narration:dict[str,Any])->dict[str,Any]:
        """Produce deterministic phrase timestamps; a real adapter replaces these with ElevenLabs timestamps."""
        _,spans=self._tts_script(narration); cursor=0.0; cues=[]
        for span in spans:
            duration=max(.35,len(span["spoken_text"])/4.2); cues.append({"scene_id":span["scene_id"],"text":span["text"],"spoken_text":span["spoken_text"],"start_seconds":round(cursor,3),"end_seconds":round(cursor+duration,3)}); cursor+=duration
        phrases=[]
        for scene_id in dict.fromkeys(c["scene_id"] for c in cues):
            group=[c for c in cues if c["scene_id"]==scene_id]; phrases.append({"scene_id":scene_id,"start_seconds":group[0]["start_seconds"],"end_seconds":group[-1]["end_seconds"],"display_text":"".join(c["text"] for c in group),"spoken_text":"".join(c["spoken_text"] for c in group)})
        return {"provider":"deterministic-timestamp-adapter","phrases":phrases,"caption_cues":cues,"duration_seconds":round(cursor,3)}

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
        """Compile a single global timeline from actual audio cue boundaries."""
        fps=self.request.output.fps; cues=alignment.get("caption_cues",[]); phrases={p["scene_id"]:p for p in alignment.get("phrases",[])}; scene_list=scenes["scenes"]
        boundaries=[0.0]
        for index in range(1,len(scene_list)):
            previous=phrases.get(scene_list[index-1]["id"]); current=phrases.get(scene_list[index]["id"])
            if previous and current: boundary=max(previous["end_seconds"],(previous["end_seconds"]+current["start_seconds"])/2)
            elif current: boundary=max(boundaries[-1],current["start_seconds"]-.2)
            else: boundary=boundaries[-1]
            boundaries.append(boundary)
        last_phrase=phrases.get(scene_list[-1]["id"]); audio_end=float(alignment.get("duration_seconds",0)); final_end=max(audio_end+.35,(last_phrase or {}).get("end_seconds",audio_end)+.35)
        if is_disclaimer_scene(scene_list[-1]): final_end=max(final_end,boundaries[-1]+4)
        boundaries.append(final_end); events=[]
        for index,scene in enumerate(scene_list):
            start_frame=round(boundaries[index]*fps); end_frame=max(start_frame,round(boundaries[index+1]*fps)-1); scene["duration_seconds"]=round((end_frame-start_frame+1)/fps,3)
            selected=scene.get("transition","cut"); transition=selected if selected in TRANSITIONS else "cut"
            events.append({"scene_id":scene["id"],"visual_kind":scene.get("visual_kind","broll"),"from_frame":start_frame,"to_frame":end_frame,"start_seconds":round(start_frame/fps,3),"end_seconds":round((end_frame+1)/fps,3),"transition":{"template":transition,"duration_frames":min(round(.6*fps),max(1,(end_frame-start_frame+1)//4))}})
        framed=[]
        for cue in cues:
            framed.append({**cue,"from_frame":round(cue["start_seconds"]*fps),"to_frame":max(round(cue["start_seconds"]*fps),round(cue["end_seconds"]*fps)-1)})
        return {"fps":fps,"total_frames":events[-1]["to_frame"]+1 if events else 0,"audio_duration_seconds":audio_end,"video_duration_seconds":round((events[-1]["to_frame"]+1)/fps,3) if events else 0,"allowed_transition_templates":sorted(TRANSITIONS),"caption_cues":framed,"events":events}

    def _assets(self,scenes:dict[str,Any])->dict[str,Any]:
        """Resolve one locked reference then apply the documented fallback ladder per scene."""
        refs={}; resolved=[]
        for scene in scenes["scenes"]:
            subject=scene.get("asset_subject")
            if subject and subject not in refs: refs[subject]={"reference_id":f"ref-{content_hash(subject)[:10]}","style_lock_id":"cartoon-v1","prompt_version":"1"}
            kind=scene.get("visual_kind","broll"); reason=f"{kind}_uses_deterministic_template" if kind!="broll" else "deterministic_mode"
            resolved.append({"scene_id":scene["id"],"strategy":"icon_illustration_gradient","attempts":[{"strategy":"icon_illustration_gradient","status":"succeeded","reason":reason}],"reference_id":refs.get(subject,{}).get("reference_id")})
        return {"reference_registry":refs,"scenes":resolved}

    def _production_assets(self,scenes:dict[str,Any])->dict[str,Any]:
        """Route ordinary B-roll to T2V and repeated subjects to Elements, with two fallbacks."""
        adapter=BachAssets(self.settings); refs={}; resolved=[]; usage={"api_calls":0,"cost_usd":0,"retries":0}
        existing_assets=self.run_dir/"assets.json"
        if existing_assets.exists():
            refs.update(json.loads(existing_assets.read_text()).get("reference_registry",{}))
        subject_counts={}
        for scene in scenes["scenes"]:
            if is_disclaimer_scene(scene) or scene.get("visual_kind","broll")!="broll":
                resolved.append({"scene_id":scene["id"],"strategy":"data_visual_template" if scene.get("visual_kind") in {"chart","metric_cards"} else "disclaimer_template","attempts":[{"strategy":"deterministic_template","status":"succeeded","reason":f"visual_kind_{scene.get('visual_kind')}"}],"reference_id":None}); continue
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
            if is_disclaimer_scene(scene) or scene.get("visual_kind","broll")!="broll": continue
            subject=scene.get("asset_subject") or scene.get("subject_id")
            prompt=scene.get("visual_prompt") or f"Cartoon-style financial explainer B-roll: {scene.get('purpose','financial scene')}, clean blue and green palette, smooth camera motion"; attempts=[]
            try:
                reference=refs.get(subject,{}).get("reference_input")
                if reference:
                    duration=max(6,min(10,round(float(scene.get("duration_seconds",6)))))
                    result,meta=adapter.elements_to_video(prompt,[reference],duration); strategy="bach_elements_to_video"
                else:
                    duration=max(6,min(10,round(float(scene.get("duration_seconds",6)))))
                    result,meta=adapter.text_to_video(prompt,duration); strategy="bach_text_to_video"
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
        summary=visual_summary(scenes,narration)
        return {"schema_version":"2.0","composition_id":"FinancialVideo","output":{"width":width,"height":height,"fps":self.request.output.fps,"codec":"h264"},"scenes":scenes["scenes"],"narration":narration,"audio":{"src":tts.get("remotion_static_src"),"duration_seconds":timeline.get("audio_duration_seconds"),"voice_settings":tts.get("voice_settings",{})},"charts":charts,"timeline":timeline,"caption_cues":timeline.get("caption_cues",[]),"assets":assets,"visual_summary":summary,"captions":{**self.request.captions.model_dump(mode="json"),"safe_zone":{"top":height-caption_height-round(height*.1),"bottom":height-round(height*.1),"left":margin,"right":width-margin}},"brand":self.request.brand.model_dump(mode="json"),"disclaimer":self.request.source_materials.disclaimer}

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
        cues=manifest.get("caption_cues",[]); events={e["scene_id"]:e for e in manifest["timeline"]["events"]}; summary=manifest["visual_summary"]
        monotonic=all(cues[i]["start_seconds"]>=cues[i-1]["end_seconds"]-1e-3 for i in range(1,len(cues)))
        within=all(c["scene_id"] in events and c["from_frame"]>=events[c["scene_id"]]["from_frame"] and c["to_frame"]<=events[c["scene_id"]]["to_frame"] for c in cues)
        narration_covered=all(segment["scene_id"] in {c["scene_id"] for c in cues} for segment in manifest["narration"]["segments"])
        chart_by_id={c["id"]:c for c in manifest["charts"]["charts"]}; data_refs=all(bool(chart_by_id.get(scene.get("chart"),{}).get("fact_ids")) for scene in manifest["scenes"] if scene.get("visual_kind") in {"chart","metric_cards"})
        audio_duration=float(manifest["audio"].get("duration_seconds") or 0); last_cue=cues[-1]["end_seconds"] if cues else 0
        checks=[{"id":"facts_traceable","passed":all(isinstance(seg.get("fact_ids"),list) for seg in manifest["narration"]["segments"])},{"id":"data_values_traceable","passed":data_refs},{"id":"transitions_template_bound","passed":all(e["transition"]["template"] in TRANSITIONS for e in events.values())},{"id":"timeline_valid","passed":all(e["to_frame"]>=e["from_frame"] for e in events.values())},{"id":"audio_content_sync","passed":abs(audio_duration-last_cue)<=.5},{"id":"caption_cues_monotonic","passed":monotonic},{"id":"caption_cues_within_scenes","passed":within},{"id":"narration_caption_coverage","passed":narration_covered},{"id":"caption_safe_zone","passed":0<=safe["left"]<safe["right"]<=width and 0<=safe["top"]<safe["bottom"]<=height},{"id":"audio_exists","passed":bool(audio_path and audio_path.exists() and audio_path.stat().st_size>0)},{"id":"data_visual_coverage","passed":summary["data_visual_coverage"]>=.8},{"id":"chart_scene_count","passed":summary["chart_count"]>=2},{"id":"metric_card_scene_count","passed":summary["metric_card_count"]>=1},{"id":"broll_share","passed":summary["broll_count"]<=math.floor(summary["content_scene_count"]*.2)},{"id":"disclaimer_present","passed":bool(manifest["disclaimer"].strip())},{"id":"disclaimer_duration","passed":is_disclaimer_scene(manifest["scenes"][-1]) and manifest["scenes"][-1]["duration_seconds"]>=4},{"id":"video_exists","passed":(self.run_dir/"final.mp4").exists() and (self.run_dir/"final.mp4").stat().st_size>1024}]
        spoken_chars=sum(len(s["spoken_text"]) for s in manifest["narration"]["segments"]); cue_chars=sum(len(c["spoken_text"]) for c in cues)
        report={"passed":all(c["passed"] for c in checks),"checks":checks,"metrics":{"audio_duration_seconds":audio_duration,"video_duration_seconds":manifest["timeline"]["video_duration_seconds"],"caption_coverage":round(cue_chars/max(1,spoken_chars),3),**summary},"manual_review":{"required":True,"keyframes_seconds":[max(0,e["from_frame"]/manifest["output"]["fps"]+.5) for e in events.values()]}}
        self._node("qa_report",report)
        if not report["passed"]: raise RuntimeError("QA failed: "+", ".join(c["id"] for c in checks if not c["passed"]))


def run_demo(job_id:str="nvidia-q1-fy27-demo",auto_approve:bool=True)->JobRecord:
    """Load the packaged NVIDIA input and execute a reproducible demo run."""
    request=JobInput.model_validate_json((ROOT/"examples/nvidia_q1_fy27/input.json").read_text())
    return Pipeline(job_id,request,auto_approve=auto_approve,settings=Settings.load("deterministic")).run()
