"""Audit biological-assembly and crystal-form compatibility of repeat controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from ospedit.rcsb_metadata import (
    cell_compatible,
    fetch_rcsb_environment_metadata,
    load_metadata_cache,
    pdb_id_from_path,
    target_profile,
)


REPORT_FORMAT = "ospedit.rcsb_environment_audit.v1"
CONTEXT_REPORT_FORMAT = "ospedit.repeat_control_context_audit.v1"


def audit_rcsb_environment(
    context_report: str | Path,
    metadata: dict[str, dict[str, Any]],
    *,
    relative_cell_length_tolerance: float = 0.1,
    cell_angle_tolerance: float = 5.0,
) -> dict[str, Any]:
    if relative_cell_length_tolerance < 0 or cell_angle_tolerance < 0:
        raise ValueError("cell tolerances must be non-negative")
    source = Path(context_report).resolve()
    context = json.loads(source.read_text())
    if context.get("format") != CONTEXT_REPORT_FORMAT:
        raise ValueError(f"expected {CONTEXT_REPORT_FORMAT}")
    rows = []
    used_entries: set[str] = set()
    for control in context.get("records", []):
        parent_id = pdb_id_from_path(control["parent"]["path"])
        repeat_id = pdb_id_from_path(control["repeat"]["path"])
        used_entries.update((parent_id, repeat_id))
        reasons = [] if control.get("selected") else ["coordinate_context_rejected"]
        parent_entry, repeat_entry = metadata.get(parent_id), metadata.get(repeat_id)
        if parent_entry is None or repeat_entry is None:
            reasons.append("missing_rcsb_entry_metadata")
        parent_profile = (
            target_profile(parent_entry, control["parent"]["target_chain"])
            if parent_entry is not None else None
        )
        repeat_profile = (
            target_profile(repeat_entry, control["repeat"]["target_chain"])
            if repeat_entry is not None else None
        )
        if parent_profile is None or repeat_profile is None:
            reasons.append("target_author_chain_not_mapped")
        parent_signatures = {
            tuple(value) for value in (parent_profile or {}).get("assembly_signatures", [])
        }
        repeat_signatures = {
            tuple(value) for value in (repeat_profile or {}).get("assembly_signatures", [])
        }
        shared_signatures = sorted(parent_signatures & repeat_signatures, key=str)
        if not parent_signatures or not repeat_signatures:
            reasons.append("target_biological_assembly_missing")
        elif not shared_signatures:
            reasons.append("biological_assembly_signature_mismatch")
        assembly_compatible = not reasons

        crystal_reasons = []
        cell_comparison = None
        if assembly_compatible:
            assert parent_entry is not None and repeat_entry is not None
            parent_space = parent_entry.get("space_group")
            repeat_space = repeat_entry.get("space_group")
            if not parent_space or not repeat_space:
                crystal_reasons.append("missing_space_group")
            elif parent_space != repeat_space:
                crystal_reasons.append("space_group_mismatch")
            cell_matches, cell_comparison = cell_compatible(
                parent_entry,
                repeat_entry,
                relative_length_tolerance=relative_cell_length_tolerance,
                angle_tolerance=cell_angle_tolerance,
            )
            if cell_comparison is None:
                crystal_reasons.append("missing_cell_metadata")
            elif not cell_matches:
                crystal_reasons.append("unit_cell_mismatch")
        else:
            crystal_reasons.append("assembly_incompatible")
        rows.append({
            "pair_id": control["pair_id"],
            "parent_id": control["parent_id"],
            "family_id": control["family_id"],
            "split": control["split"],
            "parent_pdb_id": parent_id,
            "repeat_pdb_id": repeat_id,
            "parent_profile": parent_profile,
            "repeat_profile": repeat_profile,
            "shared_assembly_signatures": [list(value) for value in shared_signatures],
            "shared_uniprot": sorted(
                set((parent_profile or {}).get("uniprot_accessions", []))
                & set((repeat_profile or {}).get("uniprot_accessions", []))
            ),
            "assembly_compatible": assembly_compatible,
            "assembly_rejection_reasons": reasons,
            "crystal_form_compatible": assembly_compatible and not crystal_reasons,
            "crystal_rejection_reasons": crystal_reasons,
            "cell_comparison": cell_comparison,
            "background": control["background"],
        })

    def summary(selected_field: str) -> dict[str, Any]:
        selected = [row for row in rows if row[selected_field]]
        return {
            "controls": len(selected),
            "pairs": len({row["pair_id"] for row in selected}),
            "families": len({row["family_id"] for row in selected}),
            "splits": {
                split: {
                    "controls": sum(row[selected_field] and row["split"] == split for row in rows),
                    "pairs": len({
                        row["pair_id"] for row in rows if row[selected_field] and row["split"] == split
                    }),
                }
                for split in ("train", "dev", "test")
            },
        }

    return {
        "format": REPORT_FORMAT,
        "context_report": str(source),
        "manifest_fingerprint": context["manifest_fingerprint"],
        "metadata_format": "ospedit.rcsb_environment_metadata.v2",
        "configuration": {
            "relative_cell_length_tolerance": relative_cell_length_tolerance,
            "cell_angle_tolerance_degree": cell_angle_tolerance,
        },
        "selection_semantics": {
            "assembly": "coordinate_context_and_shared_target_assembly_signature",
            "crystal_form": "assembly_plus_space_group_and_unit_cell_tolerance",
        },
        "limitations": [
            "assembly_signature_matches_composition_not_interface_geometry",
            "crystal_form_match_does_not_establish_identical_crystallization_conditions",
            "manual_review_is_required_before_mutation_response_attribution",
        ],
        "summary": {
            "controls": len(rows),
            "entries": len(used_entries),
            "entries_with_metadata": len(used_entries & set(metadata)),
            "assembly_compatible": summary("assembly_compatible"),
            "crystal_form_compatible": summary("crystal_form_compatible"),
        },
        "metadata": {entry_id: metadata[entry_id] for entry_id in sorted(used_entries) if entry_id in metadata},
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("context_report")
    parser.add_argument("output")
    parser.add_argument("--metadata-cache", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--relative-cell-length-tolerance", type=float, default=0.1)
    parser.add_argument("--cell-angle-tolerance", type=float, default=5.0)
    args = parser.parse_args()
    try:
        context = json.loads(Path(args.context_report).read_text())
        entry_ids = {
            pdb_id_from_path(row[side]["path"])
            for row in context.get("records", [])
            for side in ("parent", "repeat")
        }
        metadata = load_metadata_cache(args.metadata_cache)
        if not args.offline:
            metadata = fetch_rcsb_environment_metadata(
                entry_ids,
                batch_size=args.batch_size,
                timeout=args.timeout,
                retries=args.retries,
                existing=metadata,
                cache_path=args.metadata_cache,
            )
        report = audit_rcsb_environment(
            args.context_report,
            metadata,
            relative_cell_length_tolerance=args.relative_cell_length_tolerance,
            cell_angle_tolerance=args.cell_angle_tolerance,
        )
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    summary = report["summary"]
    print(
        f"RCSB environment audit written: {destination}; "
        f"assembly={summary['assembly_compatible']['controls']}/{summary['controls']}, "
        f"crystal={summary['crystal_form_compatible']['controls']}/{summary['controls']}"
    )


if __name__ == "__main__":
    main()
