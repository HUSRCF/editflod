"""Export a small, reviewable report from a student checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ospedit.data import file_sha256, json_safe
from ospedit.metrics import METRIC_SCHEMA_VERSION


def _macro_metrics(family_summary: dict[str, dict[str, float]]) -> dict[str, float]:
    if not family_summary:
        return {}
    names = sorted({name for row in family_summary.values() for name in row})
    output = {}
    for name in names:
        values = np.asarray([row.get(name, float("nan")) for row in family_summary.values()], dtype=float)
        if np.isfinite(values).any():
            output[name] = float(np.nanmean(values))
    return output


def checkpoint_report(checkpoint: str | Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise ImportError("checkpoint reporting requires torch") from error
    source = Path(checkpoint)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    config = dict(payload.get("config", {}))
    history = [float(value) for value in payload.get("history", [])]
    records = int(config.get("record_count", 0))
    batch_size = int(config.get("batch_size", 1))
    accumulation = int(config.get("grad_accumulation_steps", 1))
    steps_per_epoch = math.ceil(math.ceil(records / batch_size) / accumulation) if records else 0
    evaluation = config.get("evaluation")
    comparison = None
    if isinstance(evaluation, dict) and isinstance(evaluation.get("family_summary"), dict):
        student = _macro_metrics(evaluation["family_summary"])
        copy_payload = evaluation.get("copy_parent_baseline")
        copied = _macro_metrics(copy_payload.get("family_summary", {})) if isinstance(copy_payload, dict) else {}
        comparison = {
            "student_parent_family_macro": student,
            "copy_parent_family_macro": copied,
            "student_minus_copy": {
                name: student[name] - copied[name]
                for name in student.keys() & copied.keys()
                if np.isfinite(student[name]) and np.isfinite(copied[name])
            },
        }
    return json_safe({
        "format": "ospedit.student_checkpoint_report.v1",
        "metric_schema": METRIC_SCHEMA_VERSION,
        "checkpoint": str(source.resolve()),
        "checkpoint_sha256": file_sha256(source),
        "checkpoint_format_version": payload.get("format_version"),
        "epoch": int(payload.get("epoch", 0)),
        "optimizer_steps": int(payload.get("epoch", 0)) * steps_per_epoch,
        "loss_history": history,
        "configuration": config,
        "comparison": comparison,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    report = checkpoint_report(args.checkpoint)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
