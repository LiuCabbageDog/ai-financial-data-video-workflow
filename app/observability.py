"""Structured event and aggregate-metric recording for every pipeline node."""
from __future__ import annotations

import json, time, uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class RunLogger:
    """Append-only JSONL logger with node-level cost and latency aggregation."""
    def __init__(self, run_dir: Path, job_id: str):
        self.run_dir, self.job_id = run_dir, job_id
        self.trace_id = uuid.uuid4().hex
        self.events_path = run_dir / "events.jsonl"
        self.metrics_path = run_dir / "metrics.json"
        self.metrics: dict[str, Any] = {"job_id": job_id, "trace_id": self.trace_id, "nodes": {}, "totals": {"api_calls":0,"input_tokens":0,"output_tokens":0,"tts_characters":0,"cost_usd":0,"retries":0}}

    def event(self, event: str, **data: Any) -> None:
        """Write one correlation-rich immutable event."""
        row = {"ts": time.time(), "event": event, "job_id": self.job_id, "trace_id": self.trace_id, **data}
        with self.events_path.open("a", encoding="utf-8") as f: f.write(json.dumps(row, ensure_ascii=False)+"\n")

    @contextmanager
    def node(self, node_id: str, inputs: list[str] | None = None, scene_id: str | None = None) -> Iterator[dict[str, Any]]:
        """Measure a node and persist its inputs, outputs, usage and errors."""
        started = time.time(); usage: dict[str, Any] = {"outputs":[],"api_calls":0,"input_tokens":0,"output_tokens":0,"tts_characters":0,"cost_usd":0,"cache_hit":False,"retry":0}
        self.event("node.started", node_id=node_id, scene_id=scene_id, inputs=inputs or [])
        try:
            yield usage
            status, failure = "completed", None
        except Exception as exc:
            status, failure = "failed", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ended=time.time(); record={"status":status,"scene_id":scene_id,"started_at":started,"ended_at":ended,"latency_ms":round((ended-started)*1000,2),"inputs":inputs or [],"failure_reason":failure,**usage}
            self.metrics["nodes"][node_id if not scene_id else f"{node_id}:{scene_id}"]=record
            for key in self.metrics["totals"]:
                if key in usage: self.metrics["totals"][key]+=usage[key]
            self.metrics_path.write_text(json.dumps(self.metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            self.event(f"node.{status}", node_id=node_id, **record)
