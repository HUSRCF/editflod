"""Run multiple PreMut jobs after loading the published model once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from premut_predict import _load_upstream, _seed, predict_backbone  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a persistent PreMut model over JSON jobs")
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    jobs = json.loads(Path(args.jobs).read_text())
    if not isinstance(jobs, list) or not jobs:
        raise SystemExit("jobs must be a non-empty JSON list")
    root = Path(args.upstream_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    _seed(int(jobs[0]["seed"]), args.device)
    load_started = perf_counter()
    net = _load_upstream(root, checkpoint, args.device)
    load_seconds = perf_counter() - load_started
    rows = []
    parent_cache = {}
    for job in jobs:
        prediction, input_backbone, runtime = predict_backbone(
            net,
            root=root,
            parent=Path(job["parent_structure"]).expanduser().resolve(),
            chain=str(job["chain"]),
            mutation=str(job["mutation"]),
            device=args.device,
            seed=int(job["seed"]),
            parent_cache=parent_cache,
        )
        metadata = dict(job["metadata"])
        metadata["runtime"] = {
            "model_load_seconds_amortized": load_seconds / len(jobs),
            **runtime,
        }
        destination = Path(job["output"]).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            prediction=prediction,
            input_backbone=input_backbone,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        rows.append({"pair_id": job["pair_id"], **metadata["runtime"]})
    report = {
        "format": "ospedit.premut_batch_run.v1",
        "model_load_seconds": load_seconds,
        "parent_cache_entries": len(parent_cache),
        "jobs": rows,
    }
    destination = Path(args.report).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
