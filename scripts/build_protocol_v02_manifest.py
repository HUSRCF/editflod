"""Build the protocol-v0.2 structure-prediction dataset and frozen splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Iterable

from Bio.Align import PairwiseAligner

from ospedit.data import PairRecord, json_safe, load_manifest, manifest_fingerprint, validate_manifest, write_manifest


class _Components:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, node: str) -> str:
        self.parent.setdefault(node, node)
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _cluster_map(paths: Iterable[str | Path]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 - trusted local upstream metadata
        if not isinstance(payload, dict):
            raise ValueError(f"cluster dictionary is not a mapping: {path}")
        prefix = Path(path).name
        for cluster, members in payload.items():
            for member in members:
                token = str(member).upper()
                value = f"{prefix}:{cluster}"
                output[token].add(value)
    return dict(output)


def _structure_token(path: str, chain: str) -> str:
    return f"{Path(path).stem.upper()}_{chain.upper()}"


def _sequence_related(
    first: str,
    second: str,
    aligner: PairwiseAligner,
    *,
    min_identity: float,
    min_coverage: float,
) -> bool:
    alignment = aligner.align(first, second)[0]
    counts = alignment.counts()
    aligned = int(counts.aligned)
    shorter = min(len(first), len(second))
    return (
        shorter > 0
        and aligned / shorter >= min_coverage
        and int(counts.identities) / max(aligned, 1) >= min_identity
    )


def _balanced_assignments(
    groups: dict[str, list[PairRecord]],
    *,
    seed: int,
    train_fraction: float,
    dev_fraction: float,
) -> dict[str, str]:
    fractions = {"train": train_fraction, "dev": dev_fraction, "test": 1.0 - train_fraction - dev_fraction}
    if not 0 < train_fraction < 1 or not 0 < dev_fraction < 1 or fractions["test"] <= 0:
        raise ValueError("split fractions must all be positive")
    if len(groups) < 3:
        raise ValueError("at least three connected groups are required")
    totals = {split: 0 for split in fractions}
    target_records = {
        split: fraction * sum(len(rows) for rows in groups.values())
        for split, fraction in fractions.items()
    }
    assignments: dict[str, str] = {}
    ordered = sorted(
        groups,
        key=lambda group: (
            -len(groups[group]),
            hashlib.sha256(f"{seed}:{group}".encode()).digest(),
        ),
    )
    for index, group in enumerate(ordered):
        remaining = len(ordered) - index
        empty = [split for split, total in totals.items() if total == 0]
        candidates = empty if remaining <= len(empty) else list(fractions)
        group_size = len(groups[group])
        split = min(
            candidates,
            key=lambda name: (
                (totals[name] + group_size) / target_records[name],
                hashlib.sha256(f"{seed}:{group}:{name}".encode()).digest(),
            ),
        )
        assignments[group] = split
        totals[split] += len(groups[group])
    return assignments


def build_protocol_manifest(
    manifests: Iterable[str | Path],
    *,
    cluster_dicts: Iterable[str | Path] = (),
    min_length: int = 64,
    max_length: int = 256,
    min_sequence_identity: float = 0.5,
    min_sequence_coverage: float = 0.8,
    seed: int = 0,
    train_fraction: float = 0.7,
    dev_fraction: float = 0.15,
) -> tuple[list[PairRecord], dict[str, Any]]:
    sources = [Path(path) for path in manifests]
    if not sources:
        raise ValueError("at least one input manifest is required")
    if min_length <= 0 or max_length < min_length:
        raise ValueError("invalid length range")
    if not 0 <= min_sequence_identity <= 1 or not 0 <= min_sequence_coverage <= 1:
        raise ValueError("sequence identity and coverage must be in [0, 1]")
    records: list[PairRecord] = []
    inputs = []
    for source in sources:
        rows = load_manifest(source)
        errors = validate_manifest(rows)
        if errors:
            raise ValueError(f"input manifest failed validation ({source}): " + "; ".join(errors))
        inputs.append({"path": str(source.resolve()), "records": len(rows), "fingerprint": manifest_fingerprint(rows)})
        records.extend(rows)
    pair_ids = [record.pair.pair_id for record in records]
    if len(pair_ids) != len(set(pair_ids)):
        duplicates = sorted(pair_id for pair_id, count in Counter(pair_ids).items() if count > 1)
        raise ValueError(f"duplicate pair ids: {duplicates}")
    selected = [
        record for record in records
        if min_length <= record.pair.length <= max_length
        and len(record.pair.mutation_indices) == 1
        and record.label_source == "experimental"
    ]
    if not selected:
        raise ValueError("no records remain after protocol filtering")

    components = _Components()
    by_sequence: dict[str, list[int]] = defaultdict(list)
    token_clusters = _cluster_map(cluster_dicts)
    cluster_links = 0
    for index, record in enumerate(selected):
        row = f"row:{index}"
        components.union(row, f"parent:{record.parent_id}")
        components.union(row, f"family:{record.family_id}")
        by_sequence[record.pair.parent_sequence].append(index)
        tokens = (
            _structure_token(record.source_file, record.source_chain),
            _structure_token(record.target_file, record.target_chain),
        )
        for token in tokens:
            for cluster in token_clusters.get(token, ()):
                components.union(row, f"upstream:{cluster}")
                cluster_links += 1
    unique_sequences = list(by_sequence)
    for indices in by_sequence.values():
        for index in indices[1:]:
            components.union(f"row:{indices[0]}", f"row:{index}")

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.1
    sequence_links = 0
    for left_index, left in enumerate(unique_sequences):
        for right in unique_sequences[left_index + 1:]:
            if _sequence_related(
                left,
                right,
                aligner,
                min_identity=min_sequence_identity,
                min_coverage=min_sequence_coverage,
            ):
                components.union(f"row:{by_sequence[left][0]}", f"row:{by_sequence[right][0]}")
                sequence_links += 1

    raw_groups: dict[str, list[PairRecord]] = defaultdict(list)
    for index, record in enumerate(selected):
        raw_groups[components.find(f"row:{index}")].append(record)
    stable_groups: dict[str, list[PairRecord]] = {}
    for rows in raw_groups.values():
        digest = hashlib.sha256("\0".join(sorted(row.pair.pair_id for row in rows)).encode()).hexdigest()[:12]
        stable_groups[f"v02_family_{digest}"] = rows
    assignments = _balanced_assignments(
        stable_groups,
        seed=seed,
        train_fraction=train_fraction,
        dev_fraction=dev_fraction,
    )
    output = [
        replace(record, family_id=family, split=assignments[family])
        for family, rows in sorted(stable_groups.items())
        for record in rows
    ]
    errors = validate_manifest(output)
    if errors:
        raise ValueError("protocol manifest failed validation: " + "; ".join(errors))
    return output, json_safe({
        "format": "ospedit.protocol_v02_manifest.v1",
        "protocol": "0.2",
        "inputs": inputs,
        "input_records": len(records),
        "selected_records": len(output),
        "filtered_records": len(records) - len(output),
        "connected_groups": len(stable_groups),
        "largest_group_records": max(map(len, stable_groups.values())),
        "upstream_cluster_links": cluster_links,
        "sequence_similarity_links": sequence_links,
        "configuration": {
            "min_length": min_length,
            "max_length": max_length,
            "single_substitution": True,
            "experimental_only": True,
            "min_sequence_identity": min_sequence_identity,
            "min_sequence_coverage": min_sequence_coverage,
            "seed": seed,
            "train_fraction": train_fraction,
            "dev_fraction": dev_fraction,
        },
        "split_records": dict(Counter(record.split for record in output)),
        "split_groups": {
            split: len({record.family_id for record in output if record.split == split})
            for split in ("train", "dev", "test")
        },
        "manifest_fingerprint": manifest_fingerprint(output),
    })


def write_compact_index(records: Iterable[PairRecord], path: str | Path) -> None:
    """Write coordinate-free row provenance suitable for version control."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "pair_id", "parent_id", "family_id", "split", "length",
        "mutation_indices", "source_structure", "source_chain",
        "target_structure", "target_chain", "label_source",
    )
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "pair_id": record.pair.pair_id,
                "parent_id": record.parent_id,
                "family_id": record.family_id,
                "split": record.split,
                "length": record.pair.length,
                "mutation_indices": ",".join(map(str, record.pair.mutation_indices)),
                "source_structure": Path(record.source_file).name,
                "source_chain": record.source_chain,
                "target_structure": Path(record.target_file).name,
                "target_chain": record.target_chain,
                "label_source": record.label_source,
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--cluster-dict", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--index-output")
    parser.add_argument("--min-length", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--min-sequence-identity", type=float, default=0.5)
    parser.add_argument("--min-sequence-coverage", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    args = parser.parse_args()
    records, report = build_protocol_manifest(
        args.manifest,
        cluster_dicts=args.cluster_dict,
        min_length=args.min_length,
        max_length=args.max_length,
        min_sequence_identity=args.min_sequence_identity,
        min_sequence_coverage=args.min_sequence_coverage,
        seed=args.seed,
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
    )
    write_manifest(records, args.output)
    if args.index_output:
        write_compact_index(records, args.index_output)
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
