from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def utc_timestamp() -> str:
    # e.g. 2026-01-08T12-34-56Z (safe for filenames)
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    spec_json: Path
    module_py: Path
    metrics_json: Path
    rag_context_json: Path
    llm_transcript_json: Path


def make_run_dir(runs_root: str | Path, run_name: str) -> RunPaths:
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name.strip())[:60]
    run_dir = runs_root / f"{utc_timestamp()}_{safe}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        spec_json=run_dir / "cipher_spec.json",
        module_py=run_dir / "cipher_module.py",
        metrics_json=run_dir / "metrics.json",
        rag_context_json=run_dir / "rag_context.json",
        llm_transcript_json=run_dir / "llm_transcript.json",
    )


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
