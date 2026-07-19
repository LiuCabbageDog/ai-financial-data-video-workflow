"""FastAPI entrypoint for local job creation, review, retry and artifact browsing."""
from __future__ import annotations

import json, threading, uuid
from pathlib import Path
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from .models import JobInput, ReviewRequest, RetryRequest
from .pipeline import Pipeline, ROOT, content_hash
from .config import Settings

app = FastAPI(title="Financial Video Workflow", version="1.0.0")
_lock = threading.Lock()


def _record(job_id: str) -> dict:
    """Read a persisted job or return a standard not-found response."""
    path=Settings.load().artifact_root/job_id/"job.json"
    if not path.exists(): raise HTTPException(404,"job not found")
    return json.loads(path.read_text())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the dependency-free local operator UI."""
    return (ROOT/"web/index.html").read_text(encoding="utf-8")


@app.post("/api/jobs", status_code=202)
def create_job(request: JobInput, background: BackgroundTasks, idempotency_key: str | None=Header(default=None,alias="Idempotency-Key")) -> dict:
    """Create one asynchronous job while deduplicating repeated clicks."""
    settings=Settings.load(); settings.validate(); settings.artifact_root.mkdir(parents=True,exist_ok=True)
    digest=content_hash({"key":idempotency_key or "", "request":request.model_dump()}); index_path=settings.artifact_root/"idempotency.json"
    with _lock:
        index=json.loads(index_path.read_text()) if index_path.exists() else {}
        if digest in index: return {"job_id":index[digest],"deduplicated":True}
        job_id=f"job-{uuid.uuid4().hex[:12]}"; index[digest]=job_id; index_path.write_text(json.dumps(index,indent=2))
    pipeline=Pipeline(job_id,request,auto_approve=True,settings=settings); pipeline.save_record(); background.add_task(pipeline.run)
    return {"job_id":job_id,"deduplicated":False}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Return job, node, retry and artifact status."""
    return _record(job_id)


@app.get("/api/jobs/{job_id}/artifacts/{name}")
def get_artifact(job_id: str,name: str) -> FileResponse:
    """Serve a run artifact after preventing path traversal."""
    path=(Settings.load().artifact_root/job_id/name).resolve(); run=(Settings.load().artifact_root/job_id).resolve()
    if run not in path.parents or not path.exists(): raise HTTPException(404,"artifact not found")
    return FileResponse(path)


@app.post("/api/jobs/{job_id}/reviews/{node}/{decision}")
def review(job_id: str,node: str,decision: str,request: ReviewRequest) -> dict:
    """Persist a human approval/rejection decision for a required gate."""
    if decision not in {"approve","reject"}: raise HTTPException(400,"decision must be approve or reject")
    path=Settings.load().artifact_root/job_id/f"review_{node}.json"
    if not path.exists(): raise HTTPException(404,"review gate not found")
    path.write_text(json.dumps({"node":node,"status":"approved" if decision=="approve" else "rejected","note":request.note},ensure_ascii=False,indent=2))
    return {"ok":True,"node":node,"decision":decision}


@app.post("/api/jobs/{job_id}/scenes/{scene_id}/retry")
def retry_scene(job_id: str,scene_id: str,request: RetryRequest) -> dict:
    """Record scene-scoped invalidation without rerunning upstream facts/story."""
    run_root=Settings.load().artifact_root/job_id; record=_record(job_id); scene_plan=run_root/"scene_plan.json"
    scenes=json.loads(scene_plan.read_text())["scenes"] if scene_plan.exists() else []
    if scene_id not in {s["id"] for s in scenes}: raise HTTPException(404,"scene not found")
    retry_path=run_root/f"retry_{scene_id}_{record.get('retries',0)+1}.json"; retry_path.write_text(json.dumps({"scene_id":scene_id,"reason":request.reason,"overrides":request.overrides,"invalidated_nodes":["assets","animation_timeline","render_manifest","render","qa"]},ensure_ascii=False,indent=2))
    record["retries"]=record.get("retries",0)+1; (run_root/"job.json").write_text(json.dumps(record,ensure_ascii=False,indent=2))
    return {"ok":True,"scene_id":scene_id,"upstream_preserved":["input","canonical_facts","financial_analysis","story_plan","scene_plan"],"retry_artifact":retry_path.name}


@app.get("/api/jobs/{job_id}/events")
def events(job_id: str) -> FileResponse:
    """Expose append-only structured events (SSE adapter can tail this in production)."""
    path=Settings.load().artifact_root/job_id/"events.jsonl"
    if not path.exists(): raise HTTPException(404,"events not found")
    return FileResponse(path,media_type="application/x-ndjson")
