#!/usr/bin/env python3
"""Run the packaged NVIDIA workflow with explicit demo auto-approval."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.pipeline import run_demo

if __name__ == "__main__":
    result=run_demo()
    print(result.model_dump_json(indent=2))
