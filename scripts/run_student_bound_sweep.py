"""Run a reproducible one-pass student update-bound sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from ospedit.data import load_manifest, manifest_fingerprint
from ospedit.teacher_cache import TeacherCache


def _last_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "output" in value:
            return value
    raise ValueError("student training output did not contain a JSON result")


def _numeric_mean(rows: dict[str, Any], key: str) -> float | None:
    values = [float(summary[key]) for summary in rows.values() if isinstance(summary, dict) and summary.get(key) is not None]
    return sum(values) / len(values) if values else None


def _result_summary(result: dict[str, Any]) -> dict[str, float | None]:
    evaluation = result.get("evaluation")
    if not isinstance(evaluation, dict):
        return {"local_backbone_error": None, "mutation_site_backbone_error": None, "mutation_site_global_rmsd": None, "copy_parent_local_backbone_error": None, "copy_parent_mutation_site_backbone_error": None, "copy_parent_mutation_site_global_rmsd": None, "local_error_minus_copy": None, "mutation_site_error_minus_copy": None, "mutation_site_global_error_minus_copy": None, "parent_to_prediction_frame_rmsd": None, "distance_change_error": None, "mean_seconds": None}
    family_summary = evaluation.get("family_summary", {})
    runtime = evaluation.get("runtime_summary", {})
    local = _numeric_mean(family_summary, "local_backbone_error")
    mutation_site = _numeric_mean(family_summary, "mutation_site_backbone_error")
    mutation_site_global = _numeric_mean(family_summary, "mutation_site_global_rmsd")
    copy = evaluation.get("copy_parent_baseline", {})
    copy_family = copy.get("family_summary", {}) if isinstance(copy, dict) else {}
    copy_local = _numeric_mean(copy_family, "local_backbone_error")
    copy_mutation_site = _numeric_mean(copy_family, "mutation_site_backbone_error")
    copy_mutation_site_global = _numeric_mean(copy_family, "mutation_site_global_rmsd")
    return {
        "local_backbone_error": local,
        "mutation_site_backbone_error": mutation_site,
        "mutation_site_global_rmsd": mutation_site_global,
        "copy_parent_local_backbone_error": copy_local,
        "copy_parent_mutation_site_backbone_error": copy_mutation_site,
        "copy_parent_mutation_site_global_rmsd": copy_mutation_site_global,
        "local_error_minus_copy": local - copy_local if local is not None and copy_local is not None else None,
        "mutation_site_error_minus_copy": mutation_site - copy_mutation_site if mutation_site is not None and copy_mutation_site is not None else None,
        "mutation_site_global_error_minus_copy": mutation_site_global - copy_mutation_site_global if mutation_site_global is not None and copy_mutation_site_global is not None else None,
        "parent_to_prediction_frame_rmsd": _numeric_mean(family_summary, "parent_to_prediction_frame_rmsd"),
        "distance_change_error": _numeric_mean(family_summary, "distance_change_error"),
        "mean_seconds": float(runtime["mean_seconds"]) if "mean_seconds" in runtime else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep max-normalized-delta for the one-pass student")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bounds", type=float, nargs="+", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--eval-split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument("--mutation-loss-weight", type=float, default=0.0)
    parser.add_argument("--teacher-cache", help="Optional admitted teacher cache index.json")
    parser.add_argument("--teacher-noise-level", type=float, help="Noise level for --teacher-cache")
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--delta-loss", choices=("mse", "smooth_l1"), default="mse")
    parser.add_argument("--delta-loss-beta", type=float, default=1.0)
    parser.add_argument("--geometry-features", action="store_true")
    parser.add_argument("--no-positional-encoding", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", help="Run each bound for multiple seeds (overrides --seed)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if any(bound <= 0 for bound in args.bounds):
        parser.error("all bounds must be positive")
    if (args.teacher_cache is None) != (args.teacher_noise_level is None):
        parser.error("--teacher-cache and --teacher-noise-level must be supplied together")
    if args.distill_weight < 0:
        parser.error("--distill-weight must be non-negative")
    if args.distill_weight and args.teacher_cache is None:
        parser.error("--distill-weight requires --teacher-cache")
    seeds = tuple(args.seeds) if args.seeds else (args.seed,)
    if len(set(seeds)) != len(seeds):
        parser.error("--seeds must not contain duplicates")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(args.manifest)
    manifest_digest = manifest_fingerprint(records)
    teacher_cache_fingerprint = None
    if args.teacher_cache is not None:
        cache = TeacherCache.load(args.teacher_cache, records=records, split=args.split)
        teacher_cache_fingerprint = cache.metadata.get("manifest_fingerprint")
    runs: list[dict[str, Any]] = []
    for bound in args.bounds:
        for seed in seeds:
            suffix = f"_seed{seed}" if len(seeds) > 1 else ""
            checkpoint = output_dir / f"student_bound_{bound:g}{suffix}.pt"
            command = [
                sys.executable,
                "-m",
                "ospedit.train_cli",
                "--manifest", args.manifest,
                "--output", str(checkpoint),
                "--split", args.split,
                "--eval-split", args.eval_split,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--hidden-dim", str(args.hidden_dim),
                "--blocks", str(args.blocks),
                "--heads", str(args.heads),
                "--learning-rate", str(args.learning_rate),
                "--translation-scale", str(args.translation_scale),
                "--rotation-scale", str(args.rotation_scale),
                "--max-normalized-delta", str(bound),
                "--mutation-loss-weight", str(args.mutation_loss_weight),
                "--delta-loss", args.delta_loss,
                "--delta-loss-beta", str(args.delta_loss_beta),
                "--seed", str(seed),
                "--device", args.device,
            ]
            if args.teacher_cache is not None and args.teacher_noise_level is not None:
                command.extend(("--teacher-cache", args.teacher_cache, "--teacher-noise-level", str(args.teacher_noise_level), "--distill-weight", str(args.distill_weight)))
            if args.eval_batch_size is not None:
                command.extend(("--eval-batch-size", str(args.eval_batch_size)))
            if args.geometry_features:
                command.append("--geometry-features")
            if args.no_positional_encoding:
                command.append("--no-positional-encoding")
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            result = _last_json(completed.stdout)
            runs.append({"bound": bound, "seed": seed, "checkpoint": str(checkpoint), "summary": _result_summary(result), "result": result})

    aggregate: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for bound in args.bounds:
        matching = [run for run in runs if run["bound"] == bound]
        metrics = sorted({key for run in matching for key, value in run["summary"].items() if value is not None})
        aggregate[str(bound)] = {}
        for metric in metrics:
            values = [float(run["summary"][metric]) for run in matching if run["summary"].get(metric) is not None]
            aggregate[str(bound)][metric] = {
                "count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values)) if values else None,
            }

    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_fingerprint": manifest_digest,
        "bounds": list(args.bounds),
        "split": args.split,
        "eval_split": args.eval_split,
        "seed": args.seed,
        "seeds": list(seeds),
        "mutation_loss_weight": args.mutation_loss_weight,
        "teacher_cache": str(Path(args.teacher_cache).resolve()) if args.teacher_cache else None,
        "teacher_cache_fingerprint": teacher_cache_fingerprint,
        "teacher_noise_level": args.teacher_noise_level,
        "distill_weight": args.distill_weight,
        "delta_loss": args.delta_loss,
        "delta_loss_beta": args.delta_loss_beta,
        "geometry_features": args.geometry_features,
        "no_positional_encoding": args.no_positional_encoding,
        "aggregate": aggregate,
        "runs": runs,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"student bound sweep written: {summary_path} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
