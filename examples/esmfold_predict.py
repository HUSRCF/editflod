"""Run multiple ESMFold jobs after one model load and cache backbone outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np


BACKBONE_ATOMS = ("N", "CA", "C", "O")
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _parse_backbone(pdb: str) -> tuple[str, np.ndarray]:
    residues: list[tuple[tuple[str, str], str, dict[str, np.ndarray]]] = []
    lookup: dict[tuple[str, str], int] = {}
    for line in pdb.splitlines():
        if not line.startswith("ATOM  ") or line[21].strip() not in {"", "A"}:
            continue
        altloc = line[16].strip()
        if altloc not in {"", "A"}:
            continue
        atom = line[12:16].strip()
        if atom not in BACKBONE_ATOMS:
            continue
        key = (line[22:26].strip(), line[26].strip())
        if key not in lookup:
            lookup[key] = len(residues)
            residues.append((key, line[17:20].strip(), {}))
        residue = residues[lookup[key]][2]
        if atom not in residue:
            residue[atom] = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float
            )
    if not residues:
        raise ValueError("ESMFold PDB contains no backbone residues")
    missing = [index for index, (_, _, atoms) in enumerate(residues) if set(atoms) != set(BACKBONE_ATOMS)]
    if missing:
        raise ValueError(f"ESMFold PDB has incomplete backbone residues: {missing[:5]}")
    sequence = "".join(THREE_TO_ONE[name] for _, name, _ in residues)
    coordinates = np.asarray(
        [[atoms[name] for name in BACKBONE_ATOMS] for _, _, atoms in residues], dtype=float
    )
    return sequence, coordinates


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a persistent ESMFold model over JSON jobs")
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--esm-precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    jobs = json.loads(Path(args.jobs).read_text())
    if not isinstance(jobs, list) or not jobs:
        raise SystemExit("jobs must be a non-empty JSON list")
    import esm
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("ESMFold runner requires a visible CUDA/ROCm device")
    torch.hub.set_dir(str(Path(args.model_dir).expanduser().resolve()))
    load_started = perf_counter()
    model = esm.pretrained.esmfold_v1().eval().cuda()
    if args.esm_precision == "bf16":
        model.esm.bfloat16()
    elif args.esm_precision == "fp32":
        model.esm.float()
    model.set_chunk_size(args.chunk_size)
    torch.cuda.synchronize()
    load_seconds = perf_counter() - load_started
    rows = []
    for job in jobs:
        sequence = str(job["sequence"])
        num_recycles = job["metadata"]["num_recycles"]
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        with torch.inference_mode():
            output = model.infer([sequence], num_recycles=num_recycles)
        torch.cuda.synchronize()
        inference_seconds = perf_counter() - started
        peak_vram_bytes = int(torch.cuda.max_memory_allocated())
        conversion_started = perf_counter()
        host_output = {key: value.cpu() for key, value in output.items()}
        pdb = model.output_to_pdb(host_output)[0]
        predicted_sequence, prediction = _parse_backbone(pdb)
        if predicted_sequence != sequence:
            raise ValueError(f"ESMFold output sequence mismatch for {job['pair_id']}")
        conversion_seconds = perf_counter() - conversion_started
        metadata = dict(job["metadata"])
        metadata.update({
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "mean_plddt": float(host_output["mean_plddt"][0]),
            "ptm": float(host_output["ptm"][0]),
            "runtime": {
                "model_load_seconds_amortized": load_seconds / len(jobs),
                "inference_seconds": inference_seconds,
                "pdb_conversion_seconds": conversion_seconds,
                "peak_vram_bytes": peak_vram_bytes,
            },
        })
        destination = Path(job["output"]).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            prediction=prediction,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        rows.append({"pair_id": job["pair_id"], "mode": metadata["mode"], **metadata["runtime"]})
    report = {
        "format": "ospedit.esmfold_batch_run.v1",
        "jobs": rows,
        "model_load_seconds": load_seconds,
        "device": torch.cuda.get_device_name(0),
    }
    destination = Path(args.report).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
