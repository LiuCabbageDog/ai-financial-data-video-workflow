#!/usr/bin/env python3
"""Run the workflow in deterministic demo mode or credentialed production mode."""
from __future__ import annotations

import argparse, sys, uuid
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.config import Settings
from app.models import JobInput
from app.pipeline import Pipeline, ROOT


def main() -> None:
    """Parse CLI arguments, validate mode configuration, and launch one run."""
    parser=argparse.ArgumentParser(description="Generate a financial video")
    parser.add_argument("--mode",choices=["deterministic","production"],required=True)
    parser.add_argument("--input",help="Job input JSON. Required in production; optional in deterministic mode.")
    parser.add_argument("--job-id",default=None)
    parser.add_argument("--auto-approve",action="store_true",help="Explicitly approve the three review gates for non-interactive runs.")
    args=parser.parse_args()
    settings=Settings.load(args.mode); settings.validate()
    input_path=Path(args.input) if args.input else ROOT/"examples/nvidia_q1_fy27/input.json"
    if args.mode=="production" and not args.input: parser.error("--input is required in production mode")
    request=JobInput.model_validate_json(input_path.read_text(encoding="utf-8"))
    job_id=args.job_id or f"{args.mode}-{uuid.uuid4().hex[:12]}"
    print(Pipeline(job_id,request,auto_approve=args.auto_approve,settings=settings).run().model_dump_json(indent=2))


if __name__=="__main__": main()
