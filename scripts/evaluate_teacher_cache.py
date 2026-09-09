"""Compare admitted teacher cache deltas with experimental structure changes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from ospedit.data import load_manifest, manifest_fingerprint, validate_manifest, verify_record_checksums
from ospedit.teacher_cache import TeacherCache
from ospedit.teacher_evaluation import evaluate_teacher_cache


def _write_report(path: str | Path, payload: dict[str, object]) -> None:
    """Atomically publish a strict JSON report after evaluation completes."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate teacher delta direction against experimental pairs")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--noise-level", type=float, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--verify-checksums", action="store_true", help="verify source/target checksums before evaluation")
    parser.add_argument("--local-radius", type=float, default=10.0, help="Mutation-neighborhood radius in Angstroms")
    parser.add_argument("--remote-min-distance", type=float, default=15.0, help="Remote-region minimum distance in Angstroms")
    parser.add_argument("--min-mutation-cosine", type=float, help="Optional direction-gate lower bound")
    parser.add_argument("--max-mutation-rmse", type=float, help="Optional direction-gate upper bound")
    parser.add_argument("--max-mutation-translation-rmse", type=float, help="Optional translation RMSE upper bound in Angstroms")
    parser.add_argument("--max-mutation-rotation-rmse", type=float, help="Optional rotation RMSE upper bound in radians")
    parser.add_argument(
        "--gate-aggregation",
        choices=("record", "family", "parent"),
        default="parent",
        help="Aggregation used by direction thresholds (default: parent macro)",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for name in ("max_mutation_rmse", "max_mutation_translation_rmse", "max_mutation_rotation_rmse"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0.0):
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    records = load_manifest(args.manifest)
    if args.verify_checksums:
        errors = validate_manifest(records)
        errors.extend(error for record in records for error in verify_record_checksums(record))
        if errors:
            raise SystemExit("manifest audit failed: " + "; ".join(errors))
    selected = [record for record in records if record.split == args.split]
    cache = TeacherCache.load(args.teacher_cache, records=records, split=args.split)
    if args.local_radius < 0 or args.remote_min_distance < args.local_radius:
        parser.error("--local-radius must be nonnegative and no greater than --remote-min-distance")
    report = evaluate_teacher_cache(
        cache,
        selected,
        args.noise_level,
        local_radius=args.local_radius,
        remote_min_distance=args.remote_min_distance,
    )
    summary_key = {
        "record": "summary",
        "family": "family_macro_summary",
        "parent": "parent_macro_summary",
    }[args.gate_aggregation]
    summary = report[summary_key]
    gate_enabled = any(
        value is not None
        for value in (
            args.min_mutation_cosine,
            args.max_mutation_rmse,
            args.max_mutation_translation_rmse,
            args.max_mutation_rotation_rmse,
        )
    )
    gate_accepted = True
    gate_reasons: list[str] = []
    if args.min_mutation_cosine is not None:
        value = summary["mean_mutation_cosine"]
        if value is None or not (float(value) >= args.min_mutation_cosine):
            gate_accepted = False
            gate_reasons.append("mean mutation cosine is below threshold")
    if args.max_mutation_rmse is not None:
        value = summary["mean_mutation_rmse"]
        if value is None or not (float(value) <= args.max_mutation_rmse):
            gate_accepted = False
            gate_reasons.append("mean mutation RMSE exceeds threshold")
    for argument, key, message in (
        (args.max_mutation_translation_rmse, "mean_mutation_translation_rmse", "mean mutation translation RMSE exceeds threshold"),
        (args.max_mutation_rotation_rmse, "mean_mutation_rotation_rmse", "mean mutation rotation RMSE exceeds threshold"),
    ):
        if argument is not None:
            value = summary[key]
            if value is None or not (float(value) <= argument):
                gate_accepted = False
                gate_reasons.append(message)
    report["direction_gate"] = {
        "enabled": gate_enabled,
        "accepted": gate_accepted if gate_enabled else None,
        "min_mutation_cosine": args.min_mutation_cosine,
        "max_mutation_rmse": args.max_mutation_rmse,
        "max_mutation_translation_rmse": args.max_mutation_translation_rmse,
        "max_mutation_rotation_rmse": args.max_mutation_rotation_rmse,
        "reasons": gate_reasons,
        "aggregation": args.gate_aggregation,
        "groups": summary.get("groups", len(selected)),
    }
    report["format"] = "ospedit.teacher_evaluation.v1"
    report.update({
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_fingerprint": manifest_fingerprint(records),
        "teacher_cache": str(Path(args.teacher_cache).resolve()),
        "split": args.split,
        "local_radius": args.local_radius,
        "remote_min_distance": args.remote_min_distance,
    })
    _write_report(args.output, report)
    print(f"teacher cache evaluation written: {args.output} ({len(selected)} records)")
    if gate_enabled and not gate_accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
