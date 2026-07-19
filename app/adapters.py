"""External-provider adapters used exclusively by production mode."""
from __future__ import annotations

import base64, json, time
from pathlib import Path
from typing import Any

import fitz
import httpx

from .config import Settings


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
        schema={"type":"object","additionalProperties":False,"required":["canonical_facts","financial_analysis","story_plan","scene_plan","narration","chart_spec"],"properties":{
            "canonical_facts":{"type":"object"},"financial_analysis":{"type":"object"},"story_plan":{"type":"object"},"scene_plan":{"type":"object"},"narration":{"type":"object"},"chart_spec":{"type":"object"}}}
        system="""You are a financial-video planning engine. Use only values explicitly present in the supplied report. Canonical facts must preserve metric, base-unit value, unit, scale, currency, GAAP basis, fiscal period, calendar period end, comparison, page-level source locator, and confidence. Every numeric narration and chart claim must cite canonical fact IDs. Create 5-8 scenes and use only these transitions: cut, fade, slide, wipe, zoom. Include a final disclaimer scene of at least 4 seconds.

Follow job_input.transcript explicitly. When mode is generate, write narration in the requested language. When mode is pre-written and allow_editing is true, use transcript.text as the source script and only edit it for clarity, timing, factual consistency, and scene segmentation; do not replace it with an unrelated script. When mode is pre-written and allow_editing is false, preserve transcript.text verbatim and only segment it across scenes. Return display_text and separately normalized spoken_text for every narration segment."""
        payload={"model":self.settings.openai_model,"input":[{"role":"system","content":system},{"role":"user","content":json.dumps({"job_input":request,"parsed_reports":parsed},ensure_ascii=False)}],"text":{"format":{"type":"json_schema","name":"financial_video_plan","strict":False,"schema":schema}}}
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
        return json.loads(text),{"provider":"openai","model":self.settings.openai_model,"request_id":response.headers.get("x-request-id") or data.get("id"),"latency_ms":round((time.time()-started)*1000,2),"input_tokens":usage.get("input_tokens",0),"output_tokens":usage.get("output_tokens",0),"api_calls":1}


class ElevenLabsTTS:
    """Create production narration audio and character alignment through ElevenLabs."""
    def __init__(self,settings:Settings): self.settings=settings

    def synthesize(self,text:str,output:Path)->tuple[dict[str,Any],dict[str,Any]]:
        """Call the with-timestamps endpoint and persist the returned MP3 bytes."""
        url=f"{self.settings.elevenlabs_base_url}/text-to-speech/{self.settings.elevenlabs_voice_id}/with-timestamps"
        started=time.time()
        with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
            response=client.post(url,headers={"xi-api-key":self.settings.elevenlabs_api_key,"Content-Type":"application/json"},json={"text":text,"model_id":"eleven_multilingual_v2","output_format":"mp3_44100_128"})
            response.raise_for_status(); data=response.json()
        output.write_bytes(base64.b64decode(data["audio_base64"]))
        alignment=data.get("normalized_alignment") or data.get("alignment") or {}
        meta={"provider":"elevenlabs","voice_id":self.settings.elevenlabs_voice_id,"latency_ms":round((time.time()-started)*1000,2),"api_calls":1,"tts_characters":len(text)}
        return {"provider":"elevenlabs","artifact":output.name,"alignment":alignment},meta


class BachAssets:
    """Call configurable BACH generation endpoints and expose failures to the fallback compiler."""
    def __init__(self,settings:Settings): self.settings=settings

    def generate(self,kind:str,payload:dict[str,Any])->tuple[dict[str,Any],dict[str,Any]]:
        """Submit one video or image job using configurable paths because BACH deployments vary."""
        path=self.settings.bach_video_path if kind=="video" else self.settings.bach_image_path; started=time.time()
        with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
            response=client.post(f"{self.settings.bach_base_url}{path}",headers={"Authorization":f"Bearer {self.settings.bach_api_key}","Content-Type":"application/json"},json=payload)
            response.raise_for_status(); data=response.json()
        return data,{"provider":"bach","kind":kind,"latency_ms":round((time.time()-started)*1000,2),"api_calls":1,"request_id":response.headers.get("x-request-id") or data.get("id")}
