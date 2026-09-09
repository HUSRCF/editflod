"""Synthetic-only C0-C5 harness for checking experiment orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ospedit.data import PairRecord, StructurePair
from ospedit.experiment import (
    build_mechanism_editors,
    evaluate_editor_suite,
    evaluate_parent_workloads,
    parent_workloads,
    write_parent_workload_csv,
    write_parent_workload_report,
    write_suite_report,
)
from ospedit.geometry import residue_frames


class SyntheticEndpoint:
    """Controlled endpoint: only the edited residue gets a small response."""

    def endpoint(self, coords, sequence, noise_level, *, noise_state):
        rotations, origins, _ = residue_frames(coords, ("N", "CA", "C", "O"))
        if sequence[1] == "Y":
            origins = origins.copy()
            origins[1] += np.asarray([0.25, 0.0, 0.0])
        # A shared noise term cancels in source-target subtraction.
        origins = origins + noise_state.values[:, 1, :].mean(axis=0) * 0.01
        return rotations, origins


def make_record(mutant_aa: str = "Y", pair_id: str = "synthetic") -> PairRecord:
    residue = np.asarray([
        [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.5, 3.8, 0.0]],
        [[-1.0, 8.1, 0.0], [0.0, 7.6, 0.0], [1.0, 7.6, 0.0], [1.5, 7.6, 0.0]],
    ])
    mutant = residue.copy()
    mutant[1, :, 0] += 0.25 if mutant_aa == "Y" else 0.15
    pair = StructurePair(pair_id, "AAA", f"A{mutant_aa}A", residue, mutant, (1,), ("N", "CA", "C", "O"))
    return PairRecord(pair, "synthetic-parent", "synthetic-family", "dev")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic-only C0-C5 orchestration smoke")
    parser.add_argument("--output", default="/tmp/ospedit-mechanism-smoke.json")
    parser.add_argument("--workload-output", default="/tmp/ospedit-workload-smoke.json")
    parser.add_argument("--workload-csv-output", default="/tmp/ospedit-workload-smoke.csv")
    args = parser.parse_args()
    editors = build_mechanism_editors(endpoint_model=SyntheticEndpoint())
    records = [make_record("Y", "synthetic-y"), make_record("F", "synthetic-f")]
    suite = evaluate_editor_suite(records, editors, split="dev", batch_size=1)
    write_suite_report(suite, Path(args.output))
    workloads = parent_workloads(records, (1, 2), split="dev", seed=7)
    workload_reports = evaluate_parent_workloads(workloads, editors["C0_copy_parent"], method="C0_copy_parent")
    write_parent_workload_report(workload_reports, Path(args.workload_output))
    write_parent_workload_csv(workload_reports, Path(args.workload_csv_output))
    print(f"synthetic mechanism smoke passed: {args.output}; workloads: {args.workload_output}; csv: {args.workload_csv_output}")


if __name__ == "__main__":
    main()
