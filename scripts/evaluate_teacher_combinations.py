"""Calibrate two-level teacher-response combinations on one manifest split."""

from __future__ import annotations

import argparse
from itertools import combinations, product
import json
import math
import os
from pathlib import Path

from ospedit.data import load_manifest, manifest_fingerprint, validate_manifest, verify_record_checksums
from ospedit.teacher_cache import TeacherCache
from ospedit.teacher_evaluation import evaluate_teacher_combination


def _rank_metric(value: object, *, missing: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return missing
    return number if math.isfinite(number) else missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two-level teacher delta combinations")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--noise-levels", type=float, nargs="+", required=True)
    parser.add_argument("--weight-values", type=float, nargs="+", default=(-2.0, -1.0, -0.5, 0.5, 1.0, 2.0))
    parser.add_argument(
        "--allow-nonpositive-net-weight",
        action="store_true",
        help="Include combinations that reverse or cancel a time-invariant target-minus-source response",
    )
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--local-radius", type=float, default=10.0)
    parser.add_argument("--remote-min-distance", type=float, default=15.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    levels = tuple(dict.fromkeys(args.noise_levels))
    weights = tuple(dict.fromkeys(args.weight_values))
    if len(levels) < 2:
        parser.error("--noise-levels requires at least two distinct values")
    if not weights or any(not math.isfinite(value) for value in (*levels, *weights)):
        parser.error("noise levels and weights must be finite")
    if args.local_radius < 0 or args.remote_min_distance < args.local_radius:
        parser.error("--local-radius must be nonnegative and no greater than --remote-min-distance")
    records = load_manifest(args.manifest)
    errors = validate_manifest(records)
    if args.verify_checksums:
        errors.extend(error for record in records for error in verify_record_checksums(record))
    if errors:
        raise SystemExit("manifest audit failed: " + "; ".join(errors))
    selected = [record for record in records if record.split == args.split]
    if not selected:
        raise SystemExit(f"manifest contains no records for split={args.split!r}")
    cache = TeacherCache.load(args.teacher_cache, records=records, split=args.split)
    candidates = []
    for left, right in combinations(levels, 2):
        for left_weight, right_weight in product(weights, repeat=2):
            net_weight = left_weight + right_weight
            if net_weight <= 0 and not args.allow_nonpositive_net_weight:
                continue
            report = evaluate_teacher_combination(
                cache,
                selected,
                {left: left_weight, right: right_weight},
                local_radius=args.local_radius,
                remote_min_distance=args.remote_min_distance,
            )
            parent = report["parent_macro_summary"]
            candidates.append({
                "noise_weights": report["noise_weights"],
                "net_weight": net_weight,
                "parent_macro_summary": parent,
                "record_summary": report["summary"],
            })
    if not candidates:
        raise SystemExit("no candidate combinations remain after net-weight filtering")
    candidates.sort(
        key=lambda row: (
            -_rank_metric(row["parent_macro_summary"]["mean_mutation_cosine"], missing=-math.inf),
            _rank_metric(row["parent_macro_summary"]["mean_mutation_rmse"], missing=math.inf),
        )
    )
    payload = {
        "format": "ospedit.teacher_combination_calibration.v1",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_fingerprint": manifest_fingerprint(records),
        "teacher_cache": str(Path(args.teacher_cache).resolve()),
        "split": args.split,
        "noise_levels": list(levels),
        "weight_values": list(weights),
        "allow_nonpositive_net_weight": args.allow_nonpositive_net_weight,
        "local_radius": args.local_radius,
        "remote_min_distance": args.remote_min_distance,
        "candidates": candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, output)
    print(f"teacher combination calibration written: {output} ({len(candidates)} candidates)")


if __name__ == "__main__":
    main()
