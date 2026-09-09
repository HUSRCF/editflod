"""Evaluate the published PreMut checkpoint under ospedit's metric protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

from ospedit.data import file_sha256, load_manifest, validate_manifest, verify_record_checksums
from ospedit.experiment import (
    SuiteEvaluation,
    evaluate_editor_suite,
    write_suite_csv,
    write_suite_report,
)
from ospedit.external_baselines import PreMutEditor
from ospedit.models import CopyParentEditor


def _revision(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _python_version(executable: str | Path) -> str | None:
    completed = subprocess.run(
        [str(Path(executable).expanduser().resolve()), "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    version = (completed.stdout or completed.stderr).strip()
    return version if completed.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PreMut and copy-parent on one manifest split")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="test")
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--python", dest="python_executable", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--runner", default="examples/premut_predict.py")
    parser.add_argument("--batch-runner", default="examples/premut_batch_predict.py")
    parser.add_argument(
        "--execution-mode",
        choices=("cold_subprocess", "persistent_batch"),
        default="persistent_batch",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-reuse-cache", action="store_true")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output")
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    errors = validate_manifest(records, max_mutations=1)
    if args.verify_checksums:
        errors.extend(error for record in records for error in verify_record_checksums(record))
    selected = records if args.split == "all" else [record for record in records if record.split == args.split]
    if not selected:
        errors.append(f"manifest contains no records for split={args.split!r}")
    for record in selected:
        if len(record.pair.mutation_indices) != 1:
            errors.append(f"{record.pair.pair_id}: PreMut requires exactly one mutation")
        if not record.source_file or not record.source_chain:
            errors.append(f"{record.pair.pair_id}: source_file/source_chain are required")
    if errors:
        raise SystemExit("manifest audit failed: " + "; ".join(errors))

    root = Path(args.upstream_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    runner = Path(args.runner).expanduser().resolve()
    batch_runner = Path(args.batch_runner).expanduser().resolve()
    if not runner.is_file():
        raise SystemExit(f"--runner is not a file: {runner}")
    if args.execution_mode == "persistent_batch" and not batch_runner.is_file():
        raise SystemExit(f"--batch-runner is not a file: {batch_runner}")
    cache_runner_checksum = file_sha256(runner)
    if args.execution_mode == "persistent_batch":
        cache_runner_checksum = hashlib.sha256(
            (cache_runner_checksum + file_sha256(batch_runner)).encode()
        ).hexdigest()
    editor = PreMutEditor(
        selected,
        upstream_root=root,
        checkpoint=checkpoint,
        python_executable=args.python_executable,
        prediction_dir=args.prediction_dir,
        runner=runner,
        cache_runner_checksum=cache_runner_checksum,
        device=args.device,
        seed=args.seed,
        reuse_cache=True if args.execution_mode == "persistent_batch" else not args.no_reuse_cache,
        replay_generation_cost=args.execution_mode == "persistent_batch",
    )
    new_jobs = 0
    batch_report: Path | None = None
    if args.execution_mode == "persistent_batch":
        jobs = [
            editor.job(record)
            for record in selected
            if args.no_reuse_cache or not editor.cache_is_valid(record)
        ]
        new_jobs = len(jobs)
        if jobs:
            prediction_dir = Path(args.prediction_dir).expanduser().resolve()
            jobs_path = prediction_dir / f"premut_jobs_{args.split}_seed{args.seed}.json"
            batch_report = prediction_dir / f"premut_batch_{args.split}_seed{args.seed}.json"
            jobs_path.write_text(json.dumps(jobs, indent=2, sort_keys=True) + "\n")
            command = [
                str(Path(args.python_executable).expanduser().resolve()),
                str(batch_runner),
                "--jobs",
                str(jobs_path),
                "--upstream-root",
                str(root),
                "--checkpoint",
                str(checkpoint),
                "--device",
                args.device,
                "--report",
                str(batch_report),
            ]
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            if completed.returncode:
                detail = completed.stderr.strip() or completed.stdout.strip() or "no subprocess output"
                raise SystemExit(f"PreMut batch inference failed: {detail}")
    evaluated = evaluate_editor_suite(
        selected,
        {"C0_copy_parent": CopyParentEditor(), "B1_premut_raw": editor},
        split=None if args.split == "all" else args.split,
        run_metadata={
            "command": "run_premut_baseline",
            "manifest": str(Path(args.manifest).resolve()),
            "split": args.split,
            "upstream_root": str(root),
            "upstream_revision": _revision(root),
            "checkpoint": str(checkpoint),
            "checkpoint_checksum": file_sha256(checkpoint),
            "external_python": str(Path(args.python_executable).expanduser().resolve()),
            "external_python_version": _python_version(args.python_executable),
            "orchestrator_python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "device": args.device,
            "seed": args.seed,
            "refinement": False,
            "runtime_mode": args.execution_mode,
            "reuse_prediction_cache": not args.no_reuse_cache,
            "runner": str(runner),
            "batch_runner": str(batch_runner) if args.execution_mode == "persistent_batch" else None,
            "cache_runner_checksum": cache_runner_checksum,
            "new_prediction_jobs": new_jobs,
            "batch_report": str(batch_report) if batch_report is not None else None,
        },
    )
    suite = SuiteEvaluation(
        split=evaluated.split,
        methods=evaluated.methods,
        manifest_fingerprint=evaluated.manifest_fingerprint,
        run_metadata={**evaluated.run_metadata, "prediction_dir": str(Path(args.prediction_dir).resolve())},
    )
    write_suite_report(suite, args.output)
    if args.csv_output:
        write_suite_csv(suite, args.csv_output)
    print(json.dumps({"output": str(Path(args.output).resolve()), "records": len(selected)}))


if __name__ == "__main__":
    main()
