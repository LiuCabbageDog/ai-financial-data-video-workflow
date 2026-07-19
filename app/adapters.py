"""External-provider adapters used exclusively by production mode."""
from __future__ import annotations

import base64, hashlib, hmac, json, time
from pathlib import Path
from typing import Any

import fitz
import httpx

from .config import Settings
from .models import PlanningBundle


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

    def generate(self, request: dict[str,Any], parsed: dict[str,Any]) -> tuple[dict[str,Any],dict[str,Any]]:
        """Call OpenAI once and return the artifact bundle plus provider usage metadata."""
        schema=PlanningBundle.model_json_schema()
        system="""You are a financial-video planning engine. Use only values explicitly present in the supplied report. Canonical facts must preserve metric, base-unit value, unit, scale, currency, GAAP basis, fiscal period, calendar period end, comparison, page-level source locator, and confidence. Every numeric narration and chart claim must cite canonical fact IDs. Create 5-8 scenes and use only these transitions: cut, fade, slide, wipe, zoom. Include a final disclaimer scene of at least 4 seconds.

Follow job_input.transcript explicitly. When mode is generate, write narration in the requested language. When mode is pre-written and allow_editing is true, use transcript.text as the source script and only edit it for clarity, timing, factual consistency, and scene segmentation; do not replace it with an unrelated script. When mode is pre-written and allow_editing is false, preserve transcript.text verbatim and only segment it across scenes. Return display_text and separately normalized spoken_text for every narration segment."""
        payload={"model":self.settings.openai_model,"input":[{"role":"system","content":system},{"role":"user","content":json.dumps({"job_input":request,"parsed_reports":parsed},ensure_ascii=False)}],"text":{"format":{"type":"json_schema","name":"financial_video_plan","strict":True,"schema":schema}}}
        started=time.time()
        with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
            response=client.post(f"{self.settings.openai_base_url}/responses",headers={"Authorization":f"Bearer {self.settings.openai_api_key}","Content-Type":"application/json"},json=payload)
            response.raise_for_status(); data=response.json()
        text=data.get("output_text")
        if not text:
            chunks=[]
            for item in data.get("output",[]):
                for content in item.get("content",[]):
                    if content.get("type") in {"output_text","text"}: chunks.append(content.get("text",""))
            text="".join(chunks)
        if not text: raise RuntimeError("OpenAI response did not contain output text")
        usage=data.get("usage",{})
        bundle=PlanningBundle.model_validate_json(text)
        input_tokens=usage.get("input_tokens",0); output_tokens=usage.get("output_tokens",0); cost=input_tokens/1_000_000*self.settings.openai_input_usd_per_million+output_tokens/1_000_000*self.settings.openai_output_usd_per_million
        return bundle.model_dump(mode="json"),{"provider":"openai","model":self.settings.openai_model,"request_id":response.headers.get("x-request-id") or data.get("id"),"latency_ms":round((time.time()-started)*1000,2),"input_tokens":input_tokens,"output_tokens":output_tokens,"api_calls":1,"cost_usd":round(cost,6)}


class ElevenLabsTTS:
    """Create production narration audio and character alignment through ElevenLabs."""
    def __init__(self,settings:Settings): self.settings=settings

    def synthesize(self,text:str,output:Path)->tuple[dict[str,Any],dict[str,Any]]:
        """Call the with-timestamps endpoint and persist the returned MP3 bytes."""
        url=f"{self.settings.elevenlabs_base_url}/text-to-speech/{self.settings.elevenlabs_voice_id}/with-timestamps"
        started=time.time()
        with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
            response=client.post(
                url,
                headers={"xi-api-key":self.settings.elevenlabs_api_key,"Content-Type":"application/json"},
                params={"output_format":self.settings.elevenlabs_output_format},
                json={"text":text,"model_id":self.settings.elevenlabs_model_id},
            )
            response.raise_for_status(); data=response.json()
        output.write_bytes(base64.b64decode(data["audio_base64"]))
        alignment=data.get("normalized_alignment") or data.get("alignment") or {}
        meta={"provider":"elevenlabs","voice_id":self.settings.elevenlabs_voice_id,"model_id":self.settings.elevenlabs_model_id,"output_format":self.settings.elevenlabs_output_format,"latency_ms":round((time.time()-started)*1000,2),"api_calls":1,"tts_characters":len(text),"cost_usd":round(len(text)/1000*self.settings.elevenlabs_usd_per_thousand_chars,6)}
        return {"provider":"elevenlabs","artifact":output.name,"alignment":alignment},meta


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

    def text_to_video(self,prompt:str)->tuple[dict[str,Any],dict[str,Any]]:
        """Generate ordinary B-roll without reference assets."""
        return self._generate("video",self.settings.bach_text_to_video_path,{"model_name":self.settings.bach_model_name,"prompt":prompt,"resolution":self.settings.bach_resolution,"aspect_ratio":"16:9","duration":6,"fps":30,"generate_audio":False},"video_url")

    def elements_to_video(self,prompt:str,reference_inputs:list[dict[str,Any]])->tuple[dict[str,Any],dict[str,Any]]:
        """Generate consistency-sensitive B-roll from BACH subject/image references."""
        return self._generate("video",self.settings.bach_elements_to_video_path,{"reference_inputs":reference_inputs,"prompt":prompt,"resolution":self.settings.bach_resolution,"duration":6,"aspect_ratio":"16:9","fps":24,"generate_audio":False},"video_url")

    def text_to_image(self,prompt:str)->tuple[dict[str,Any],dict[str,Any]]:
        """Generate one 16:9 fallback still for Remotion camera motion."""
        return self._generate("image",self.settings.bach_text_to_image_path,{"prompt":prompt,"output_count":1,"aspect_ratio":"16:9","image_size":"2K","quality":"medium","output_mime_type":"image/png"},"image_urls")

    def text_to_subject(self,name:str,description:str,subject_type:str="character")->tuple[dict[str,Any],dict[str,Any]]:
        """Generate and lock a four-view cartoon subject for reuse across scenes."""
        return self._generate("image",self.settings.bach_text_to_subject_path,{"name":name[:50],"description":description[:1000],"style":"cartoon","subject_type":subject_type},"split_images")
