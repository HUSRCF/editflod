"""Build a diagnostic manual-review queue from environment-audited controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ospedit.data import load_manifest, manifest_fingerprint, validate_manifest


REPORT_FORMAT = "ospedit.response_review_queue.v1"
ENDPOINT_FORMAT = "ospedit.rcsb_endpoint_environment_audit.v1"
CONTROL_FORMAT = "ospedit.rcsb_environment_audit.v1"
SIGNAL_FORMAT = "ospedit.background_signal_diagnostic.v1"


def build_response_review_queue(
    manifest: str | Path,
    endpoint_report: str | Path,
    control_report: str | Path,
    signal_report: str | Path,
    *,
    min_controls: int = 2,
    min_local_sbr: float = 1.0,
    min_distance_sbr: float = 1.0,
) -> dict[str, Any]:
    if min_controls <= 0 or min_local_sbr < 0 or min_distance_sbr < 0:
        raise ValueError("control count must be positive and S/B thresholds non-negative")
    records = load_manifest(manifest)
    errors = validate_manifest(records)
    if errors:
        raise ValueError("manifest audit failed: " + "; ".join(errors))
    fingerprint = manifest_fingerprint(records)
    inputs = {}
    for name, path, expected in (
        ("endpoint", endpoint_report, ENDPOINT_FORMAT),
        ("control", control_report, CONTROL_FORMAT),
        ("signal", signal_report, SIGNAL_FORMAT),
    ):
        resolved = Path(path).resolve()
        payload = json.loads(resolved.read_text())
        if payload.get("format") != expected:
            raise ValueError(f"{name} report must use {expected}")
        if payload.get("manifest_fingerprint") != fingerprint:
            raise ValueError(f"{name} report uses a different manifest")
        inputs[name] = payload
    by_pair = {record.pair.pair_id: record for record in records}
    endpoints = {row["pair_id"]: row for row in inputs["endpoint"]["records"]}
    controls: dict[str, list[dict[str, Any]]] = {}
    for row in inputs["control"]["records"]:
        if row.get("crystal_form_compatible"):
            controls.setdefault(row["pair_id"], []).append(row)
    signals = {
        row["pair_id"]: row
        for row in inputs["signal"].get("rcsb_crystal_form_records", [])
    }
    endpoint_metadata = inputs["endpoint"].get("metadata", {})
    control_metadata = inputs["control"].get("metadata", {})

    def entry_review(metadata: dict[str, Any], pdb_id: str) -> dict[str, Any] | None:
        entry = metadata.get(pdb_id)
        if entry is None:
            return None
        return {
            "title": entry.get("title"),
            "primary_citation": entry.get("primary_citation"),
            "experimental_method": entry.get("experimental_method"),
            "resolution_angstrom": entry.get("resolution_angstrom"),
            "space_group": entry.get("space_group"),
            "crystal_growth": entry.get("crystal_growth"),
        }
    candidates = []
    for pair_id, pair_controls in sorted(controls.items()):
        endpoint = endpoints.get(pair_id)
        signal = signals.get(pair_id)
        if endpoint is None or signal is None or not endpoint.get("crystal_form_compatible"):
            continue
        if len(pair_controls) < min_controls:
            continue
        local_sbr = signal["local"]["signal_to_background_max"]
        distance_sbr = signal["distance"]["signal_to_background_max"]
        if local_sbr is None or distance_sbr is None:
            continue
        if float(local_sbr) < min_local_sbr or float(distance_sbr) < min_distance_sbr:
            continue
        record = by_pair[pair_id]
        mutation = [
            {
                "index": index,
                "source": record.pair.parent_sequence[index],
                "target": record.pair.mutant_sequence[index],
            }
            for index in record.pair.mutation_indices
        ]
        candidates.append({
            "pair_id": pair_id,
            "parent_id": record.parent_id,
            "family_id": record.family_id,
            "split": record.split,
            "mutation": mutation,
            "parent_pdb_id": endpoint["parent_pdb_id"],
            "mutant_pdb_id": endpoint["mutant_pdb_id"],
            "parent_entry": entry_review(endpoint_metadata, endpoint["parent_pdb_id"]),
            "mutant_entry": entry_review(endpoint_metadata, endpoint["mutant_pdb_id"]),
            "endpoint_shared_assembly_signatures": endpoint["shared_assembly_signatures"],
            "endpoint_shared_uniprot": endpoint["shared_uniprot"],
            "endpoint_cell_comparison": endpoint["cell_comparison"],
            "endpoint_shared_primary_citation_ids": endpoint.get(
                "shared_primary_citation_ids", []
            ),
            "endpoint_primary_citation_matched": endpoint.get(
                "primary_citation_matched", False
            ),
            "endpoint_shared_crystal_growth_signatures": endpoint.get(
                "shared_crystal_growth_signatures", []
            ),
            "endpoint_declared_crystal_growth_matched": endpoint.get(
                "declared_crystal_growth_matched", False
            ),
            "repeat_controls": [
                {
                    "pdb_id": control["repeat_pdb_id"],
                    "entry": entry_review(control_metadata, control["repeat_pdb_id"]),
                    "shared_assembly_signatures": control["shared_assembly_signatures"],
                    "shared_uniprot": control["shared_uniprot"],
                    "cell_comparison": control["cell_comparison"],
                    "background": control["background"],
                }
                for control in pair_controls
            ],
            "repeat_control_count": len(pair_controls),
            "signal_to_background_max": {
                region: signal[region]["signal_to_background_max"]
                for region in ("local", "site", "distance")
            },
            "signal_to_background_median": {
                region: signal[region]["signal_to_background_median"]
                for region in ("local", "site", "distance")
            },
        })
    return {
        "format": REPORT_FORMAT,
        "manifest": str(Path(manifest).resolve()),
        "manifest_fingerprint": fingerprint,
        "endpoint_report": str(Path(endpoint_report).resolve()),
        "control_report": str(Path(control_report).resolve()),
        "signal_report": str(Path(signal_report).resolve()),
        "usage": "manual_review_queue_not_training_or_test_data",
        "configuration": {
            "mutation_indexing": "zero_based_internal_sequence_index",
            "require_endpoint_crystal_form_compatible": True,
            "require_control_crystal_form_compatible": True,
            "min_controls": min_controls,
            "min_local_signal_to_max_background": min_local_sbr,
            "min_distance_signal_to_max_background": min_distance_sbr,
        },
        "limitations": [
            "selection_uses_mutant_endpoint_signal_and_must_not_be_an_inference_feature",
            "crystal_form_metadata_does_not_replace_manual_experimental_context_review",
            "shared_citation_or_declared_growth_is_reported_as_evidence_not_a_selection_rule",
            "queue_is_too_small_and_split_incomplete_for_model_evaluation",
        ],
        "summary": {
            "candidates": len(candidates),
            "splits": {
                split: sum(row["split"] == split for row in candidates)
                for split in ("train", "dev", "test")
            },
            "families": len({row["family_id"] for row in candidates}),
        },
        "records": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("endpoint_report")
    parser.add_argument("control_report")
    parser.add_argument("signal_report")
    parser.add_argument("output")
    parser.add_argument("--min-controls", type=int, default=2)
    parser.add_argument("--min-local-sbr", type=float, default=1.0)
    parser.add_argument("--min-distance-sbr", type=float, default=1.0)
    args = parser.parse_args()
    try:
        report = build_response_review_queue(
            args.manifest,
            args.endpoint_report,
            args.control_report,
            args.signal_report,
            min_controls=args.min_controls,
            min_local_sbr=args.min_local_sbr,
            min_distance_sbr=args.min_distance_sbr,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"response review queue written: {destination} ({len(report['records'])} candidates)")


if __name__ == "__main__":
    main()
