"""External-provider adapters used exclusively by production mode."""
from __future__ import annotations

import base64, hashlib, hmac, json, time
from pathlib import Path
from typing import Any

import fitz
import httpx

from .config import Settings
from .models import ProviderNarration, ProviderPlanningBundle


def parse_financial_reports(paths: list[str]) -> dict[str, Any]:
    """Extract page-addressable text from local PDF reports using PyMuPDF."""
    documents=[]
    for raw in paths:
        path=Path(raw).expanduser().resolve()
        if not path.exists(): raise FileNotFoundError(f"Financial report not found: {path}")
        if path.suffix.lower() != ".pdf": raise ValueError(f"Production financial report must be a PDF: {path}")
        with fitz.open(path) as pdf:
            pages=[{"page":i+1,"text":page.get_text("text")} for i,page in enumerate(pdf)]
        documents.append({"path":str(path),"filename":path.name,"pages":pages})
    return {"documents":documents}


class OpenAIPlanner:
    """Generate a schema-constrained production planning bundle with the Responses API."""
    def __init__(self, settings: Settings): self.settings=settings

    def _response(self,payload:dict[str,Any])->tuple[str,dict[str,Any],str|None,float]:
        """Execute one Responses request and expose useful provider errors."""
        started=time.time()
        with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
            response=client.post(f"{self.settings.openai_base_url}/responses",headers={"Authorization":f"Bearer {self.settings.openai_api_key}","Content-Type":"application/json"},json=payload)
            try: response.raise_for_status()
            except httpx.HTTPStatusError as error:
                detail=response.text.strip()[:2000] or "empty response body"
                raise RuntimeError(f"OpenAI Responses API returned HTTP {response.status_code}: {detail}") from error
            data=response.json()
        text=data.get("output_text") or "".join(content.get("text","") for item in data.get("output",[]) for content in item.get("content",[]) if content.get("type") in {"output_text","text"})
        if not text: raise RuntimeError("OpenAI response did not contain output text")
        return text,data.get("usage",{}),response.headers.get("x-request-id") or data.get("id"),round((time.time()-started)*1000,2)

    def generate(self, request: dict[str,Any], parsed: dict[str,Any]) -> tuple[dict[str,Any],dict[str,Any]]:
        """Generate a data-first plan and iteratively rewrite narration to its duration budget."""
        system="""You are a financial-video planning engine. Use only values explicitly present in the supplied report. Represent every numeric fact with the appropriate typed quantity and preserve source metadata. Narration uses text/fact parts: text parts contain no digits, currency symbols, or percent signs, and every numeric claim uses a fact part. Charts contain only fact-backed series.

Create six to eight scenes. Exactly one final disclaimer scene has kind=disclaimer, visual_kind=disclaimer, and at least four planned seconds. For content scenes, at least eighty percent use visual_kind chart or metric_cards, at least two use chart, at least one uses metric_cards, and at most twenty percent use broll. Data scenes reference a matching chart; metric_cards uses chart type metric_cards. B-roll is only for an opening, transition, or non-chartable business explanation. Use only cut, fade, slide, wipe, zoom transitions.

Choose chart types from the meaning of the data, not from a preferred default. Use donut for composition, share, contribution, or percentage-of-total stories, and give it mutually exclusive parts rather than a total plus one subset; use line for growth-rate comparisons or a metric across ordered time periods; range for low/mid/high guidance; horizontal_bar for many categories or long labels; waterfall for additive financial bridges; gauge for one percentage KPI; metric_cards for two to four headline KPIs; table for simple listings, mixed units, or values that should not share an axis; bar only for a small number of comparable categories with the same unit. Never put money, percentages, and per-share values on one shared axis. Avoid using bar for most scenes. Keep opening B-roll narration concise enough for an eight-second visual.

For generated Mandarin narration, stay within the requested total video duration, including the final disclaimer. Target the configured Mandarin character budget supplied in job_input and prioritize substantive analysis over padding. Follow pre-written transcript editing rules exactly. Use null for unavailable nullable fields and empty arrays where appropriate; return every schema field."""
        target=(request.get("content_requirements") or {}).get("target_duration_seconds")
        if target:
            request={**request,"duration_budget":{"target_seconds":target,"mandarin_chars_per_second":self.settings.mandarin_chars_per_second,"planning_tolerance":self.settings.narration_planning_tolerance,"acceptance_tolerance":self.settings.target_duration_tolerance}}
        payload={"model":self.settings.openai_model,"input":[{"role":"system","content":system},{"role":"user","content":json.dumps({"job_input":request,"parsed_reports":parsed},ensure_ascii=False)}],"text":{"format":{"type":"json_schema","name":"financial_video_plan","strict":True,"schema":ProviderPlanningBundle.model_json_schema()}}}
        text,usage,request_id,latency=self._response(payload)
        provider=ProviderPlanningBundle.model_validate_json(text); bundle=provider.to_domain(); repair_count=0; request_ids=[request_id] if request_id else []
        estimate=sum(len(s.spoken_text) for s in bundle.narration.segments)/self.settings.mandarin_chars_per_second
        tolerance=self.settings.narration_planning_tolerance
        outside_budget=bool(target and not float(target)*(1-tolerance)<=estimate<=float(target)*(1+tolerance))
        estimates=[round(estimate,2)]
        while request.get("transcript",{}).get("mode","generate")=="generate" and target and outside_budget and repair_count<self.settings.narration_rewrite_attempts:
            minimum=round(float(target)*(1-tolerance)*self.settings.mandarin_chars_per_second); maximum=round(float(target)*(1+tolerance)*self.settings.mandarin_chars_per_second)
            context={"canonical_facts":provider.canonical_facts.model_dump(mode="json"),"financial_analysis":provider.financial_analysis.model_dump(mode="json"),"story_plan":provider.story_plan.model_dump(mode="json"),"scene_plan":[s.model_dump(mode="json") for s in provider.scene_plan],"existing_narration":provider.narration.model_dump(mode="json")}
            ratio=float(target)/max(estimate,.1)
            direction="compress" if estimate>float(target) else "expand"
            repair={"model":self.settings.openai_model,"input":[{"role":"system","content":f"Rewrite only narration. Preserve all scene IDs and use only supplied fact IDs. Return ProviderNarration JSON. Text parts cannot contain digits, currency symbols, or percent signs; numeric claims must be fact parts. The current compiled narration is estimated at {estimate:.1f} seconds and must {direction} toward {float(target):.1f} seconds (length ratio {ratio:.3f}). Produce {minimum} to {maximum} spoken Chinese characters total, including the disclaimer. Remove repetition and secondary commentary before dropping essential facts. Keep the opening scene at no more than eight seconds of speech. Do not add facts or scenes."},{"role":"user","content":json.dumps(context,ensure_ascii=False)}],"text":{"format":{"type":"json_schema","name":"financial_video_narration","strict":True,"schema":ProviderNarration.model_json_schema()}}}
            repair_text,repair_usage,repair_id,repair_latency=self._response(repair)
            provider.narration=ProviderNarration.model_validate_json(repair_text); bundle=provider.to_domain(); repair_count+=1
            usage={"input_tokens":usage.get("input_tokens",0)+repair_usage.get("input_tokens",0),"output_tokens":usage.get("output_tokens",0)+repair_usage.get("output_tokens",0)}
            if repair_id: request_ids.append(repair_id)
            latency+=repair_latency
            estimate=sum(len(s.spoken_text) for s in bundle.narration.segments)/self.settings.mandarin_chars_per_second
            estimates.append(round(estimate,2)); outside_budget=not float(target)*(1-tolerance)<=estimate<=float(target)*(1+tolerance)
        if request.get("transcript",{}).get("mode","generate")=="generate" and target and outside_budget:
            raise RuntimeError(f"Narration remains outside target duration after {repair_count} rewrite attempts: estimated {estimate:.1f}s, target {float(target):.1f}s ±{tolerance:.0%}; estimates={estimates}")
        input_tokens=usage.get("input_tokens",0); output_tokens=usage.get("output_tokens",0); cost=input_tokens/1_000_000*self.settings.openai_input_usd_per_million+output_tokens/1_000_000*self.settings.openai_output_usd_per_million
        return bundle.model_dump(mode="json"),{"provider":"openai","model":self.settings.openai_model,"request_id":",".join(request_ids) or None,"latency_ms":latency,"input_tokens":input_tokens,"output_tokens":output_tokens,"api_calls":1+repair_count,"narration_rewritten":repair_count>0,"narration_rewrite_count":repair_count,"duration_estimates_seconds":estimates,"estimated_audio_seconds":round(estimate,2),"target_duration_seconds":target,"planning_tolerance":tolerance,"acceptance_tolerance":self.settings.target_duration_tolerance,"cost_usd":round(cost,6)}


class ElevenLabsTTS:
    """Create production narration audio and character alignment through ElevenLabs."""
    def __init__(self,settings:Settings): self.settings=settings

    def synthesize(self,text:str,output:Path,speed:float=.95,voice_id:str|None=None)->tuple[dict[str,Any],dict[str,Any]]:
        """Call the with-timestamps endpoint and persist the returned MP3 bytes."""
        selected_voice_id=voice_id or self.settings.elevenlabs_voice_id
        if not selected_voice_id: raise ValueError("ElevenLabs voice_id is required")
        url=f"{self.settings.elevenlabs_base_url}/text-to-speech/{selected_voice_id}/with-timestamps"
        started=time.time()
        with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
            response=client.post(
                url,
                headers={"xi-api-key":self.settings.elevenlabs_api_key,"Content-Type":"application/json"},
                params={"output_format":self.settings.elevenlabs_output_format},
                json={"text":text,"model_id":self.settings.elevenlabs_model_id,"voice_settings":{"stability":self.settings.elevenlabs_stability,"similarity_boost":self.settings.elevenlabs_similarity_boost,"style":self.settings.elevenlabs_style,"use_speaker_boost":self.settings.elevenlabs_use_speaker_boost,"speed":speed}},
            )
            response.raise_for_status(); data=response.json()
        output.write_bytes(base64.b64decode(data["audio_base64"]))
        # `normalized_alignment` may transliterate Mandarin into spaced pinyin,
        # so its array indices no longer match character offsets in `text`.
        alignment=data.get("alignment") or data.get("normalized_alignment") or {}
        voice_settings={"stability":self.settings.elevenlabs_stability,"similarity_boost":self.settings.elevenlabs_similarity_boost,"style":self.settings.elevenlabs_style,"use_speaker_boost":self.settings.elevenlabs_use_speaker_boost,"speed":speed}
        meta={"provider":"elevenlabs","voice_id":selected_voice_id,"model_id":self.settings.elevenlabs_model_id,"output_format":self.settings.elevenlabs_output_format,"voice_settings":voice_settings,"latency_ms":round((time.time()-started)*1000,2),"api_calls":1,"tts_characters":len(text),"cost_usd":round(len(text)/1000*self.settings.elevenlabs_usd_per_thousand_chars,6)}
        return {"provider":"elevenlabs","artifact":output.name,"alignment":alignment,"voice_id":selected_voice_id,"voice_settings":voice_settings},meta


class BachAssets:
    """Implement BACH's signed asynchronous image, subject, and video contracts."""
    def __init__(self,settings:Settings): self.settings=settings

    @staticmethod
    def _b64url(value:bytes)->str:
        """Encode one JWT segment without padding, as required by RFC 7519."""
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

    def _token(self,now:int|None=None)->str:
        """Create BACH's HS256 bearer token from the server-side AccessKey and SecretKey."""
        issued=int(time.time() if now is None else now)
        header=self._b64url(json.dumps({"alg":"HS256","typ":"JWT"},separators=(",",":")).encode())
        claims=self._b64url(json.dumps({"iss":self.settings.bach_access_key,"nbf":issued-5,"exp":issued+864000},separators=(",",":")).encode())
        signing_input=f"{header}.{claims}".encode()
        signature=self._b64url(hmac.new((self.settings.bach_secret_key or "").encode(),signing_input,hashlib.sha256).digest())
        return f"{header}.{claims}.{signature}"

    @staticmethod
    def _data(response:httpx.Response)->dict[str,Any]:
        """Validate BACH's business envelope and return its nested data object."""
        response.raise_for_status(); envelope=response.json()
        if envelope.get("code") != 200: raise RuntimeError(f"BACH API error: {envelope.get('message') or envelope.get('code')}")
        data=envelope.get("data")
        if not isinstance(data,dict): raise RuntimeError("BACH response is missing data")
        return data

    def _generate(self,kind:str,path:str,payload:dict[str,Any],result_field:str)->tuple[dict[str,Any],dict[str,Any]]:
        """Submit a task, poll the same endpoint by task ID, and require its media result."""
        started=time.time(); calls=0
        with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
            headers={"Authorization":f"Bearer {self._token()}","Content-Type":"application/json"}
            response=client.post(f"{self.settings.bach_base_url}{path}",headers=headers,json=payload)
            data=self._data(response); calls=1
            task_id=data.get("task_id")
            if not task_id: raise RuntimeError("BACH asynchronous response is missing data.task_id")
            status=str(data.get("status") or data.get("message") or "TASK_PENDING").upper()
            while status in {"TASK_PENDING","TASK_PROCESSING"}:
                if time.time()-started > self.settings.request_timeout_seconds: raise TimeoutError("BACH task timed out")
                time.sleep(self.settings.bach_poll_interval_seconds)
                data=self._data(client.get(f"{self.settings.bach_base_url}{path}/{task_id}",headers=headers)); calls+=1
                status=str(data.get("status") or data.get("message") or "").upper()
            if status != "TASK_SUCCEEDED": raise RuntimeError(f"BACH task failed: {data.get('message') or data.get('reason') or status}")
            if not data.get(result_field): raise RuntimeError(f"BACH result is missing data.{result_field}")
        unit_cost=self.settings.bach_video_usd_per_call if kind=="video" else self.settings.bach_image_usd_per_call
        return data,{"provider":"bach","kind":kind,"endpoint":path,"task_id":task_id,"latency_ms":round((time.time()-started)*1000,2),"api_calls":calls,"cost_usd":round(unit_cost,6),"request_id":response.headers.get("x-request-id") or task_id}

    def text_to_video(self,prompt:str,duration:int=6)->tuple[dict[str,Any],dict[str,Any]]:
        """Generate ordinary B-roll without reference assets."""
        return self._generate("video",self.settings.bach_text_to_video_path,{"model_name":self.settings.bach_model_name,"prompt":prompt,"resolution":self.settings.bach_resolution,"aspect_ratio":"16:9","duration":duration,"fps":30,"generate_audio":False},"video_url")

    def elements_to_video(self,prompt:str,reference_inputs:list[dict[str,Any]],duration:int=6)->tuple[dict[str,Any],dict[str,Any]]:
        """Generate consistency-sensitive B-roll from BACH subject/image references."""
        return self._generate("video",self.settings.bach_elements_to_video_path,{"reference_inputs":reference_inputs,"prompt":prompt,"resolution":self.settings.bach_resolution,"duration":duration,"aspect_ratio":"16:9","fps":24,"generate_audio":False},"video_url")

    def text_to_image(self,prompt:str)->tuple[dict[str,Any],dict[str,Any]]:
        """Generate one 16:9 fallback still for Remotion camera motion."""
        return self._generate("image",self.settings.bach_text_to_image_path,{"prompt":prompt,"output_count":1,"aspect_ratio":"16:9","image_size":"2K","quality":"medium","output_mime_type":"image/png"},"image_urls")

    def text_to_subject(self,name:str,description:str,subject_type:str="character")->tuple[dict[str,Any],dict[str,Any]]:
        """Generate and lock a four-view cartoon subject for reuse across scenes."""
        return self._generate("image",self.settings.bach_text_to_subject_path,{"name":name[:50],"description":description[:1000],"style":"cartoon","subject_type":subject_type},"split_images")
