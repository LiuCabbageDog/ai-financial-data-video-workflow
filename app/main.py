"""FastAPI entrypoint for local job creation, review, retry and artifact browsing."""
from __future__ import annotations

import fcntl, json, threading, uuid
from pathlib import Path
from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, UploadFile
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
    request_hash=content_hash(request.model_dump(mode="json")); digest=content_hash({"key":idempotency_key}) if idempotency_key else request_hash; index_path=settings.artifact_root/"idempotency.json"
    with _lock:
        with index_path.open("a+",encoding="utf-8") as index_file:
            fcntl.flock(index_file,fcntl.LOCK_EX); index_file.seek(0); raw=index_file.read(); index=json.loads(raw) if raw else {}; found=index.get(digest)
            if found:
                entry={"job_id":found,"request_hash":request_hash} if isinstance(found,str) else found
                if entry["request_hash"]!=request_hash: raise HTTPException(409,"Idempotency-Key was already used with a different request")
                return {"job_id":entry["job_id"],"deduplicated":True}
            job_id=f"job-{uuid.uuid4().hex[:12]}"; index[digest]={"job_id":job_id,"request_hash":request_hash}; index_file.seek(0); index_file.truncate(); index_file.write(json.dumps(index,indent=2)); index_file.flush()
    pipeline=Pipeline(job_id,request,auto_approve=False,settings=settings); pipeline.save_record(); background.add_task(pipeline.run)
    return {"job_id":job_id,"deduplicated":False}


@app.post("/api/uploads",status_code=201)
async def upload_report(report: UploadFile=File(...)) -> dict:
    """Store one PDF in the local artifact input area and return its path."""
    if not report.filename or Path(report.filename).suffix.lower()!=".pdf": raise HTTPException(400,"only PDF reports are supported")
    data=await report.read()
    if not data.startswith(b"%PDF-"): raise HTTPException(400,"uploaded file is not a valid PDF")
    root=Settings.load().artifact_root/"uploads"; root.mkdir(parents=True,exist_ok=True); target=root/f"{uuid.uuid4().hex}-{Path(report.filename).name}"
    target.write_bytes(data); return {"path":str(target),"filename":report.filename,"size":len(data)}


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
def review(job_id: str,node: str,decision: str,request: ReviewRequest,background:BackgroundTasks) -> dict:
    """Persist a human approval/rejection decision for a required gate."""
    if decision not in {"approve","reject"}: raise HTTPException(400,"decision must be approve or reject")
    if node not in {"story_plan","narration","chart_spec"}: raise HTTPException(400,"unknown review node")
    path=Settings.load().artifact_root/job_id/f"review_{node}.json"
    if not path.exists(): raise HTTPException(404,"review gate not found")
    path.write_text(json.dumps({"node":node,"status":"approved" if decision=="approve" else "rejected","note":request.note,"reviewer":"local-operator"},ensure_ascii=False,indent=2))
    if decision=="approve":
        run=path.parent; job_input=JobInput.model_validate_json((run/"input.json").read_text()); pipeline=Pipeline(job_id,job_input,auto_approve=False,settings=Settings.load()); background.add_task(pipeline.run)
    return {"ok":True,"node":node,"decision":decision}


@app.post("/api/jobs/{job_id}/scenes/{scene_id}/retry",status_code=202)
def retry_scene(job_id: str,scene_id: str,request: RetryRequest,background:BackgroundTasks) -> dict:
    """Record scene-scoped invalidation without rerunning upstream facts/story."""
    run_root=Settings.load().artifact_root/job_id; record=_record(job_id); scene_plan=run_root/"scene_plan.json"
    scenes=json.loads(scene_plan.read_text())["scenes"] if scene_plan.exists() else []
    if scene_id not in {s["id"] for s in scenes}: raise HTTPException(404,"scene not found")
    retry_path=run_root/f"retry_{scene_id}_{record.get('retries',0)+1}.json"; retry_path.write_text(json.dumps({"scene_id":scene_id,"reason":request.reason,"overrides":request.overrides,"invalidated_nodes":["assets","animation_timeline","render_manifest","render","qa"],"status":"queued"},ensure_ascii=False,indent=2))
    record["retries"]=record.get("retries",0)+1; record["status"]="queued"; record["current_node"]=f"retry:{scene_id}"; (run_root/"job.json").write_text(json.dumps(record,ensure_ascii=False,indent=2))
    job_input=JobInput.model_validate_json((run_root/"input.json").read_text()); pipeline=Pipeline(job_id,job_input,settings=Settings.load()); background.add_task(pipeline.retry_scene,scene_id,request.overrides,retry_path)
    return {"ok":True,"scene_id":scene_id,"upstream_preserved":["input","canonical_facts","financial_analysis","story_plan","scene_plan"],"retry_artifact":retry_path.name}


@app.get("/api/jobs/{job_id}/events")
def events(job_id: str) -> FileResponse:
    """Expose append-only structured events (SSE adapter can tail this in production)."""
    path=Settings.load().artifact_root/job_id/"events.jsonl"
    if not path.exists(): raise HTTPException(404,"events not found")
    return FileResponse(path,media_type="application/x-ndjson")
