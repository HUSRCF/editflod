"""Run a fixed-budget multi-seed comparison of one-pass student architectures."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np

from ospedit.data import file_sha256, json_safe, load_manifest, manifest_fingerprint


REPORT_FORMAT = "ospedit.student_architecture_sweep.v1"
ARCHITECTURES = ("transformer", "spatial_graph")
PRIMARY_METRICS = (
    "local_backbone_error",
    "mutation_site_backbone_error",
    "distance_change_error",
    "distance_change_cosine",
    "local_distance_change_error",
    "local_distance_change_cosine",
    "remote_target_error",
    "remote_scaffold_drift",
    "predicted_distance_change_norm",
    "true_distance_change_norm",
)


def _last_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "output" in payload:
            return payload
    raise ValueError("student training output did not contain a JSON result")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"command failed ({completed.returncode}): {message}")
    return completed


def _family_macro(families: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for metric in PRIMARY_METRICS:
        values = np.asarray(
            [family.get(metric, float("nan")) for family in families.values()],
            dtype=float,
        )
        output[metric] = (
            float(np.nanmean(values)) if np.isfinite(values).any() else None
        )
    return output


def _optional_difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _comparison(
    student_families: dict[str, dict[str, Any]],
    copy_families: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    student = _family_macro(student_families)
    copied = _family_macro(copy_families)
    return {
        "student_family_macro": student,
        "copy_parent_family_macro": copied,
        "student_minus_copy": {
            metric: _optional_difference(student[metric], copied[metric])
            for metric in PRIMARY_METRICS
        },
    }


def aggregate_runs(runs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(runs)
    aggregate: dict[str, Any] = {}
    for architecture in sorted({str(run["architecture"]) for run in materialized}):
        matching = [run for run in materialized if run["architecture"] == architecture]
        architecture_summary: dict[str, Any] = {
            "runs": len(matching),
            "parameter_count": sorted({int(run["parameter_count"]) for run in matching}),
        }
        for split in ("train", "dev"):
            split_summary: dict[str, Any] = {}
            for section in (
                "student_family_macro",
                "copy_parent_family_macro",
                "student_minus_copy",
            ):
                section_summary: dict[str, Any] = {}
                for metric in PRIMARY_METRICS:
                    values = np.asarray(
                        [
                            run[split][section].get(metric, float("nan"))
                            for run in matching
                        ],
                        dtype=float,
                    )
                    finite = values[np.isfinite(values)]
                    section_summary[metric] = {
                        "count": int(len(finite)),
                        "mean": float(np.mean(finite)) if len(finite) else None,
                        "std": float(np.std(finite)) if len(finite) else None,
                    }
                split_summary[section] = section_summary
            architecture_summary[split] = split_summary
        aggregate[architecture] = architecture_summary
    return aggregate


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument(
        "--architectures", nargs="+", choices=ARCHITECTURES, default=ARCHITECTURES
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--spatial-neighbors", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--mutation-loss-weight", type=float, default=0.0)
    parser.add_argument("--neighborhood-loss-weight", type=float, default=0.0)
    parser.add_argument("--local-distance-loss-weight", type=float, default=0.0)
    parser.add_argument("--mutation-vector-loss-weight", type=float, default=0.0)
    parser.add_argument("--biochemical-edit-features", action="store_true")
    parser.add_argument("--max-normalized-delta", type=float)
    parser.add_argument("--target-localization-radius", type=float)
    parser.add_argument("--target-localization-transition", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument(
        "--within-family-probe",
        action="store_true",
        help="Permit family/parent overlap for a diagnostic endpoint holdout",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch size must be positive")
    if args.hidden_dim <= 0 or args.blocks <= 0 or args.heads <= 0:
        parser.error("model dimensions must be positive")
    if args.learning_rate <= 0 or args.gradient_clip_norm <= 0:
        parser.error("learning rate and gradient clip norm must be positive")
    if (
        args.mutation_loss_weight < 0
        or args.neighborhood_loss_weight < 0
        or args.local_distance_loss_weight < 0
        or args.mutation_vector_loss_weight < 0
    ):
        parser.error("regional loss weights must be non-negative")
    if args.max_normalized_delta is not None and args.max_normalized_delta <= 0:
        parser.error("max normalized delta must be positive")
    if args.target_localization_radius is not None and args.target_localization_radius <= 0:
        parser.error("target localization radius must be positive")
    if args.target_localization_transition <= 0:
        parser.error("target localization transition must be positive")
    if len(set(args.architectures)) != len(args.architectures):
        parser.error("architectures must not contain duplicates")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must not contain duplicates")

    records = load_manifest(args.manifest)
    splits = {record.split for record in records}
    if not {"train", "dev"} <= splits:
        raise SystemExit("manifest must contain both train and dev records")
    if "test" in splits:
        raise SystemExit("architecture sweep manifest must not contain frozen test records")
    train_count = sum(record.split == "train" for record in records)
    dev_count = sum(record.split == "dev" for record in records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_dev_path = output_dir / "copy_parent_dev.json"
    copy_command = [
        sys.executable,
        "-m",
        "ospedit.cli",
        "--manifest",
        args.manifest,
        "--eval-split",
        "dev",
        "--editor",
        "copy_source_backbone",
        "--batch-size",
        str(args.batch_size),
        "--results-output",
        str(copy_dev_path),
    ]
    if args.within_family_probe:
        copy_command.append("--allow-split-overlap")
    _run(copy_command)
    copy_dev = _load_json(copy_dev_path)

    runs = []
    for architecture in args.architectures:
        for seed in args.seeds:
            stem = f"{architecture}_seed{seed}"
            checkpoint = output_dir / f"{stem}.pt"
            train_command = [
                sys.executable,
                "-m",
                "ospedit.train_cli",
                "--manifest",
                args.manifest,
                "--output",
                str(checkpoint),
                "--split",
                "train",
                "--eval-split",
                "train",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--eval-batch-size",
                str(args.batch_size),
                "--student-architecture",
                architecture,
                "--hidden-dim",
                str(args.hidden_dim),
                "--blocks",
                str(args.blocks),
                "--heads",
                str(args.heads),
                "--spatial-neighbors",
                str(args.spatial_neighbors),
                "--learning-rate",
                str(args.learning_rate),
                "--gradient-clip-norm",
                str(args.gradient_clip_norm),
                "--mutation-loss-weight",
                str(args.mutation_loss_weight),
                "--neighborhood-loss-weight",
                str(args.neighborhood_loss_weight),
                "--local-distance-loss-weight",
                str(args.local_distance_loss_weight),
                "--mutation-vector-loss-weight",
                str(args.mutation_vector_loss_weight),
                "--family-balanced-loss",
                "--endpoint-group-balanced-loss",
                "--seed",
                str(seed),
                "--device",
                args.device,
            ]
            if args.max_normalized_delta is not None:
                train_command.extend(
                    ("--max-normalized-delta", str(args.max_normalized_delta))
                )
            if args.target_localization_radius is not None:
                train_command.extend((
                    "--target-localization-radius",
                    str(args.target_localization_radius),
                    "--target-localization-transition",
                    str(args.target_localization_transition),
                ))
            if args.verify_checksums:
                train_command.append("--verify-checksums")
            if args.within_family_probe:
                train_command.append("--allow-split-overlap")
            if args.biochemical_edit_features:
                train_command.append("--biochemical-edit-features")
            started = time.perf_counter()
            train_result = _last_json(_run(train_command).stdout)
            train_command_seconds = time.perf_counter() - started

            dev_path = output_dir / f"{stem}_dev.json"
            dev_command = [
                sys.executable,
                "-m",
                "ospedit.cli",
                "--manifest",
                args.manifest,
                "--eval-split",
                "dev",
                "--editor",
                "student",
                "--student-checkpoint",
                str(checkpoint),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
                "--results-output",
                str(dev_path),
            ]
            if args.within_family_probe:
                dev_command.append("--allow-split-overlap")
            _run(dev_command)
            dev_result = _load_json(dev_path)

            try:
                import torch
            except ImportError as error:  # pragma: no cover
                raise SystemExit("architecture sweep requires torch") from error
            checkpoint_payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            parameter_count = sum(
                int(value.numel())
                for value in checkpoint_payload["model_state"].values()
            )
            history = [float(value) for value in checkpoint_payload.get("history", [])]
            train_evaluation = train_result["evaluation"]
            runs.append({
                "architecture": architecture,
                "seed": seed,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
                "dev_report": str(dev_path.resolve()),
                "dev_report_sha256": file_sha256(dev_path),
                "parameter_count": parameter_count,
                "epochs": int(checkpoint_payload["epoch"]),
                "optimizer_steps": int(checkpoint_payload["epoch"])
                * math.ceil(train_count / args.batch_size),
                "train_command_seconds_including_evaluation": train_command_seconds,
                "loss_history": history,
                "loss_initial": history[0] if history else None,
                "loss_final": history[-1] if history else None,
                "loss_minimum": min(history) if history else None,
                "train": _comparison(
                    train_evaluation["family_summary"],
                    train_evaluation["copy_parent_baseline"]["family_summary"],
                ),
                "dev": _comparison(
                    dev_result["family_summary"], copy_dev["family_summary"]
                ),
            })
            print(
                f"completed {architecture} seed={seed}: "
                f"loss={runs[-1]['loss_final']:.6g}",
                flush=True,
            )

    try:
        import torch

        torch_version = torch.__version__
    except ImportError:  # pragma: no cover
        torch_version = None
    report = json_safe({
        "format": REPORT_FORMAT,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_fingerprint": manifest_fingerprint(records),
        "selection_used_observed_response": False,
        "frozen_test_records_processed": 0,
        "purpose": "fixed_budget_development_architecture_comparison",
        "selection_policy": {
            "primary_metric": "dev_family_macro_local_backbone_error",
            "copy_parent_noninferiority_required": True,
            "secondary_metrics": [
                "mutation_site_backbone_error",
                "distance_change_error",
                "distance_change_cosine",
                "local_distance_change_error",
                "local_distance_change_cosine",
                "remote_target_error",
                "remote_scaffold_drift",
            ],
            "test_metrics_used": False,
        },
        "configuration": {
            "architectures": list(args.architectures),
            "seeds": list(args.seeds),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "blocks": args.blocks,
            "heads": args.heads,
            "spatial_neighbors": args.spatial_neighbors,
            "learning_rate": args.learning_rate,
            "gradient_clip_norm": args.gradient_clip_norm,
            "mutation_loss_weight": args.mutation_loss_weight,
            "neighborhood_loss_weight": args.neighborhood_loss_weight,
            "local_distance_loss_weight": args.local_distance_loss_weight,
            "mutation_vector_loss_weight": args.mutation_vector_loss_weight,
            "biochemical_edit_features": args.biochemical_edit_features,
            "max_normalized_delta": args.max_normalized_delta,
            "target_localization_radius": args.target_localization_radius,
            "target_localization_transition": args.target_localization_transition,
            "family_balanced_loss": True,
            "endpoint_group_balanced_loss": True,
            "teacher_distillation": False,
            "device": args.device,
            "verify_checksums": args.verify_checksums,
            "within_family_probe": args.within_family_probe,
            "train_records": train_count,
            "dev_records": dev_count,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch_version,
        },
        "runs": runs,
        "aggregate": aggregate_runs(runs),
    })
    destination = Path(args.summary_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"architecture sweep written: {destination} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
