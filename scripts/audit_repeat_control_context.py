"""Pre-screen repeat controls for comparable coordinate-file environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from ospedit.structure_context import structure_context


REPORT_FORMAT = "ospedit.repeat_control_context_audit.v1"
BACKGROUND_REPORT_FORMAT = "ospedit.background_control_coverage.v1"


def audit_repeat_control_context(
    background_report: str | Path,
    *,
    ignored_hetero: Iterable[str] = ("HOH",),
    protein_contact_radius: float = 5.0,
    hetero_contact_radius: float = 6.0,
    max_resolution_difference: float | None = None,
) -> dict[str, Any]:
    if protein_contact_radius <= 0 or hetero_contact_radius <= 0:
        raise ValueError("contact radii must be positive")
    if max_resolution_difference is not None and max_resolution_difference < 0:
        raise ValueError("max_resolution_difference must be non-negative")
    source = Path(background_report).resolve()
    background = json.loads(source.read_text())
    if background.get("format") != BACKGROUND_REPORT_FORMAT:
        raise ValueError(f"expected {BACKGROUND_REPORT_FORMAT}")
    ignored = frozenset(value.strip().upper() for value in ignored_hetero if value.strip())
    cache: dict[tuple[str, str], dict[str, Any]] = {}

    def context(path: str, chain: str) -> dict[str, Any]:
        key = str(Path(path).resolve()), chain
        if key not in cache:
            cache[key] = structure_context(
                key[0],
                chain,
                ignored_hetero=ignored,
                protein_contact_radius=protein_contact_radius,
                hetero_contact_radius=hetero_contact_radius,
            )
        return cache[key]

    rows: list[dict[str, Any]] = []
    for pair in background.get("records", []):
        parent = context(pair["parent_structure"], pair["parent_chain"])
        for control in pair["controls"]:
            repeat = context(control["repeat_structure"], control["repeat_chain"])
            reasons: list[str] = []
            if parent["proximal_hetero"] != repeat["proximal_hetero"]:
                reasons.append("proximal_hetero_mismatch")
            parent_has_contact = parent["protein_contact_chain_count"] > 0
            repeat_has_contact = repeat["protein_contact_chain_count"] > 0
            if parent_has_contact != repeat_has_contact:
                reasons.append("protein_contact_presence_mismatch")
            if parent["structure_method"] != repeat["structure_method"]:
                reasons.append("structure_method_mismatch")
            resolution_difference = None
            if parent["resolution_angstrom"] is not None and repeat["resolution_angstrom"] is not None:
                resolution_difference = abs(
                    float(parent["resolution_angstrom"]) - float(repeat["resolution_angstrom"])
                )
                if (
                    max_resolution_difference is not None
                    and resolution_difference > max_resolution_difference
                ):
                    reasons.append("resolution_difference_exceeds_limit")
            rows.append({
                "pair_id": pair["pair_id"],
                "parent_id": pair["parent_id"],
                "family_id": pair["family_id"],
                "split": pair["split"],
                "repeat_structure": control["repeat_structure"],
                "repeat_chain": control["repeat_chain"],
                "background": control["background"],
                "parent": parent,
                "repeat": repeat,
                "resolution_difference_angstrom": resolution_difference,
                "selected": not reasons,
                "rejection_reasons": reasons,
            })
    selected_pair_ids = {row["pair_id"] for row in rows if row["selected"]}
    split_summary = {}
    for split in ("train", "dev", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        split_summary[split] = {
            "controls": len(split_rows),
            "selected_controls": sum(row["selected"] for row in split_rows),
            "pairs": len({row["pair_id"] for row in split_rows}),
            "pairs_with_selected_control": len({
                row["pair_id"] for row in split_rows if row["selected"]
            }),
        }
    return {
        "format": REPORT_FORMAT,
        "background_report": str(source),
        "manifest_fingerprint": background["manifest_fingerprint"],
        "configuration": {
            "ignored_hetero": sorted(ignored),
            "protein_contact_radius_angstrom": protein_contact_radius,
            "hetero_contact_radius_angstrom": hetero_contact_radius,
            "max_resolution_difference_angstrom": max_resolution_difference,
        },
        "selection_semantics": "coordinate_file_context_prescreen_only",
        "biological_assembly_verified": False,
        "limitations": [
            "asymmetric_unit_contacts_do_not_establish_biological_assembly",
            "matching_ligand_names_do_not_establish_matching_occupancy_or_binding_state",
            "crystal_form_and_construct_equivalence_require_metadata_review",
        ],
        "summary": {
            "controls": len(rows),
            "selected_controls": sum(row["selected"] for row in rows),
            "pairs": len({row["pair_id"] for row in rows}),
            "pairs_with_selected_control": len(selected_pair_ids),
            "splits": split_summary,
        },
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("background_report")
    parser.add_argument("output")
    parser.add_argument("--ignore-hetero", nargs="*", default=("HOH",))
    parser.add_argument("--protein-contact-radius", type=float, default=5.0)
    parser.add_argument("--hetero-contact-radius", type=float, default=6.0)
    parser.add_argument("--max-resolution-difference", type=float)
    args = parser.parse_args()
    try:
        report = audit_repeat_control_context(
            args.background_report,
            ignored_hetero=args.ignore_hetero,
            protein_contact_radius=args.protein_contact_radius,
            hetero_contact_radius=args.hetero_contact_radius,
            max_resolution_difference=args.max_resolution_difference,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    summary = report["summary"]
    print(
        f"repeat-control context audit written: {destination} "
        f"({summary['selected_controls']}/{summary['controls']} controls selected)"
    )


if __name__ == "__main__":
    main()
