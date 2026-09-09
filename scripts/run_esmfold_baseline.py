"""Evaluate default and zero-extra-recycling ESMFold under ospedit metrics."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys

from ospedit.data import file_sha256, load_manifest, validate_manifest, verify_record_checksums
from ospedit.experiment import evaluate_editor_suite, write_suite_csv, write_suite_report
from ospedit.external_baselines import ESMFoldEditor
from ospedit.models import CopyParentEditor


MODEL_FILES = (
    "esmfold_3B_v1.pt",
    "esm2_t36_3B_UR50D.pt",
    "esm2_t36_3B_UR50D-contact-regression.pt",
)


def _model_identity(model_dir: Path) -> dict[str, str]:
    checkpoint_dir = model_dir / "checkpoints"
    identity = {}
    for name in MODEL_FILES:
        path = checkpoint_dir / name
        if not path.is_file():
            raise ValueError(f"ESMFold cache is missing {path}")
        identity[name] = file_sha256(path)
    return identity


def _python_version(executable: Path) -> str | None:
    completed = subprocess.run(
        [str(executable), "--version"], text=True, capture_output=True, check=False
    )
    version = (completed.stdout or completed.stderr).strip()
    return version if completed.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ESMFold baselines on a manifest split")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="test")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--python", dest="python_executable", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--runner", default="examples/esmfold_predict.py")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--esm-precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("default", "zero_extra_recycles"),
        default=("default", "zero_extra_recycles"),
    )
    parser.add_argument("--no-reuse-cache", action="store_true")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output")
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    errors = validate_manifest(records)
    if args.verify_checksums:
        errors.extend(error for record in records for error in verify_record_checksums(record))
    selected = records if args.split == "all" else [record for record in records if record.split == args.split]
    if not selected:
        errors.append(f"manifest contains no records for split={args.split!r}")
    if errors:
        raise SystemExit("manifest audit failed: " + "; ".join(errors))

    model_dir = Path(args.model_dir).expanduser().resolve()
    external_python = Path(args.python_executable).expanduser().resolve()
    runner = Path(args.runner).expanduser().resolve()
    prediction_dir = Path(args.prediction_dir).expanduser().resolve()
    prediction_dir.mkdir(parents=True, exist_ok=True)
    if not external_python.is_file() or not runner.is_file():
        raise SystemExit("--python and --runner must be existing files")
    try:
        model_identity = _model_identity(model_dir)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    runner_checksum = file_sha256(runner)
    editors = {
        mode: ESMFoldEditor(
            prediction_dir=prediction_dir,
            mode=mode,
            model_identity=model_identity,
            runner_checksum=runner_checksum,
            chunk_size=args.chunk_size,
            esm_precision=args.esm_precision,
        )
        for mode in args.modes
    }
    jobs = [
        editor.job(record.pair)
        for editor in editors.values()
        for record in selected
        if args.no_reuse_cache or not editor.cache_is_valid(record.pair)
    ]
    batch_report = prediction_dir / f"esmfold_batch_{args.split}.json"
    if jobs:
        jobs_path = prediction_dir / f"esmfold_jobs_{args.split}.json"
        jobs_path.write_text(json.dumps(jobs, indent=2, sort_keys=True) + "\n")
        command = [
            str(external_python),
            str(runner),
            "--jobs",
            str(jobs_path),
            "--model-dir",
            str(model_dir),
            "--chunk-size",
            str(args.chunk_size),
            "--esm-precision",
            args.esm_precision,
            "--report",
            str(batch_report),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no subprocess output"
            raise SystemExit(f"ESMFold batch inference failed: {detail}")

    suite = evaluate_editor_suite(
        selected,
        {
            "C0_copy_parent": CopyParentEditor(),
            **{f"B2_esmfold_{mode}": editor for mode, editor in editors.items()},
        },
        split=None if args.split == "all" else args.split,
        run_metadata={
            "command": "run_esmfold_baseline",
            "manifest": str(Path(args.manifest).expanduser().resolve()),
            "split": args.split,
            "model_dir": str(model_dir),
            "model_identity": model_identity,
            "external_python": str(external_python),
            "external_python_version": _python_version(external_python),
            "orchestrator_python_version": sys.version.split()[0],
            "runner": str(runner),
            "runner_checksum": runner_checksum,
            "prediction_dir": str(prediction_dir),
            "batch_report": str(batch_report) if jobs else None,
            "new_prediction_jobs": len(jobs),
            "modes": list(args.modes),
            "chunk_size": args.chunk_size,
            "esm_precision": args.esm_precision,
            "platform": platform.platform(),
            "alignment": "global_backbone_kabsch_to_parent",
        },
    )
    write_suite_report(suite, args.output)
    if args.csv_output:
        write_suite_csv(suite, args.csv_output)
    print(json.dumps({"output": str(Path(args.output).resolve()), "records": len(selected), "jobs": len(jobs)}))


if __name__ == "__main__":
    main()
