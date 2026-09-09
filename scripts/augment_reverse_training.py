"""Add reverse directions for experimental training endpoints."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Iterable

from ospedit.data import (
    PairRecord,
    StructurePair,
    json_safe,
    load_manifest,
    manifest_fingerprint,
    write_manifest,
)
from ospedit.endpoint_groups import endpoint_group_key, endpoint_identity


REPORT_FORMAT = "ospedit.reverse_training_augmentation.v1"


def _direction(record: PairRecord) -> tuple[str, str]:
    return endpoint_identity(record, source=True), endpoint_identity(record, source=False)


def _edit(record: PairRecord) -> str:
    return ",".join(
        f"{record.pair.parent_sequence[index]}>{record.pair.mutant_sequence[index]}"
        for index in record.pair.mutation_indices
    )


def _reverse_record(record: PairRecord) -> PairRecord:
    pair = StructurePair(
        pair_id=f"{record.pair.pair_id}__reverse_train",
        parent_sequence=record.pair.mutant_sequence,
        mutant_sequence=record.pair.parent_sequence,
        parent_coords=record.pair.mutant_coords.copy(),
        mutant_coords=record.pair.parent_coords.copy(),
        mutation_indices=record.pair.mutation_indices,
        atom_names=record.pair.atom_names,
    )
    environment = dict(record.environment_metadata)
    environment["training_augmentation"] = {
        "kind": "reverse_experimental_endpoint",
        "source_pair_id": record.pair.pair_id,
        "independent_evidence": False,
    }
    return replace(
        record,
        pair=pair,
        parent_id=f"reverse:{record.parent_id}:{record.pair.pair_id}",
        source_file=record.target_file,
        target_file=record.source_file,
        source_chain=record.target_chain,
        target_chain=record.source_chain,
        source_checksum=record.target_checksum,
        target_checksum=record.source_checksum,
        environment_metadata=environment,
    )


def augment_reverse_training(
    records: Iterable[PairRecord],
) -> tuple[list[PairRecord], dict[str, Any]]:
    source = list(records)
    pair_ids = {record.pair.pair_id for record in source}
    directions = {_direction(record) for record in source if record.split == "train"}
    augmented: list[PairRecord] = []
    skipped_existing = 0
    for record in source:
        if record.split != "train" or record.label_source != "experimental":
            continue
        left, right = _direction(record)
        if (right, left) in directions:
            skipped_existing += 1
            continue
        reverse = _reverse_record(record)
        if reverse.pair.pair_id in pair_ids:
            raise ValueError(f"reverse pair_id collision: {reverse.pair.pair_id}")
        pair_ids.add(reverse.pair.pair_id)
        augmented.append(reverse)

    output = source + augmented
    output.sort(key=lambda record: (record.split, record.family_id, record.pair.pair_id))
    before_edits = Counter(_edit(record) for record in source if record.split == "train")
    after_edits = Counter(_edit(record) for record in output if record.split == "train")
    report = {
        "format": REPORT_FORMAT,
        "source_manifest_fingerprint": manifest_fingerprint(source),
        "augmented_manifest_fingerprint": manifest_fingerprint(output),
        "selection_used_observed_response": False,
        "teacher_labels_added": 0,
        "dev_or_test_records_augmented": 0,
        "summary": {
            "source_records": len(source),
            "source_train_records": sum(record.split == "train" for record in source),
            "reverse_train_records_added": len(augmented),
            "existing_reverse_records_skipped": skipped_existing,
            "output_train_records": sum(record.split == "train" for record in output),
            "output_dev_records": sum(record.split == "dev" for record in output),
            "directed_edit_types_before": len(before_edits),
            "directed_edit_types_after": len(after_edits),
            "physical_endpoint_groups_before": len(
                {endpoint_group_key(record) for record in source}
            ),
            "physical_endpoint_groups_after": len(
                {endpoint_group_key(record) for record in output}
            ),
        },
        "directed_edit_counts_before": before_edits,
        "directed_edit_counts_after": after_edits,
        "augmented_pairs": [
            {
                "pair_id": record.pair.pair_id,
                "source_pair_id": record.environment_metadata["training_augmentation"][
                    "source_pair_id"
                ],
                "family_id": record.family_id,
                "directed_edit": _edit(record),
            }
            for record in augmented
        ],
        "limitations": [
            "reverse_directions_are_training_augmentation_not_independent_evidence",
            "reverse_directions_do_not_add_new_physical_endpoint_pairs",
            "endpoint_group_balancing_remains_required",
        ],
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output_manifest")
    parser.add_argument("output_report")
    args = parser.parse_args()

    output, report = augment_reverse_training(load_manifest(args.manifest))
    write_manifest(output, args.output_manifest)
    destination = Path(args.output_report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        f"reverse training augmentation written: {args.output_manifest} "
        f"({report['summary']['reverse_train_records_added']} added)"
    )


if __name__ == "__main__":
    main()
