from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ospedit.data import PairRecord, pair_parsed_structures, parse_structure, write_manifest
from scripts.audit_repeat_structures import audit_repeat_pairs


def _write_structure(path: Path, middle: str = "ALA", middle_y_shift: float = 0.0) -> None:
    lines = []
    serial = 1
    for residue_index, residue_name in enumerate(("ALA", middle, "ALA"), start=1):
        base = float(residue_index * 3)
        for atom_name, x_offset, y_offset in (
            ("N", -1.0, 0.5),
            ("CA", 0.0, 0.0),
            ("C", 1.0, 0.0),
            ("O", 1.5, 0.8),
        ):
            y_shift = middle_y_shift if residue_index == 2 else 0.0
            lines.append(
                f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} A{residue_index:4d}"
                f"    {base + x_offset:8.3f}{0.2 * residue_index + y_offset + y_shift:8.3f}{0.0:8.3f}"
                "  1.00 20.00           C\n"
            )
            serial += 1
    path.write_text("".join(lines) + "END\n")


def test_repeat_structure_audit_reports_zero_for_identical_structures(tmp_path):
    _write_structure(tmp_path / "parent.pdb")
    _write_structure(tmp_path / "repeat.pdb")
    pairs = tmp_path / "repeats.csv"
    pairs.write_text(
        "pair_id,parent_structure,parent_chain,repeat_structure,repeat_chain,mutation_index\n"
        "repeat-1,parent.pdb,A,repeat.pdb,A,1\n"
    )

    report = audit_repeat_pairs(pairs)

    assert report["format"] == "ospedit.repeat_structure_audit.v3"
    assert report["records"][0]["mapping"] == "residue_id"
    assert report["records"][0]["backbone_rmsd_angstrom"] == pytest.approx(0.0, abs=1e-7)
    assert report["records"][0]["max_normalized_delta_norm"] == pytest.approx(0.0, abs=1e-7)


def test_repeat_structure_audit_rejects_sequence_mismatch(tmp_path):
    _write_structure(tmp_path / "parent.pdb")
    _write_structure(tmp_path / "repeat.pdb", middle="TYR")
    pairs = tmp_path / "repeats.csv"
    pairs.write_text(
        "pair_id,parent_structure,parent_chain,repeat_structure,repeat_chain,mutation_index\n"
        "repeat-1,parent.pdb,A,repeat.pdb,A,1\n"
    )

    with pytest.raises(ValueError, match="identical sequences"):
        audit_repeat_pairs(pairs)


def test_repeat_structure_audit_uses_conservative_max_across_repeats(tmp_path):
    _write_structure(tmp_path / "parent.pdb")
    _write_structure(tmp_path / "repeat-small.pdb", middle_y_shift=0.1)
    _write_structure(tmp_path / "repeat-large.pdb", middle_y_shift=0.6)
    pairs = tmp_path / "repeats.csv"
    pairs.write_text(
        "pair_id,parent_structure,parent_chain,repeat_structure,repeat_chain,mutation_index\n"
        "repeat-1,parent.pdb,A,repeat-small.pdb,A,1\n"
        "repeat-1,parent.pdb,A,repeat-large.pdb,A,1\n"
    )

    report = audit_repeat_pairs(
        pairs,
        background_aggregation="max",
        min_repeat_structures=2,
    )

    raw_backgrounds = [row["neighborhood_rmsd_angstrom"] for row in report["records"]]
    assert report["pairs"][0]["repeat_structures"] == 2
    assert report["pairs"][0]["neighborhood_rmsd_angstrom"] == pytest.approx(
        max(raw_backgrounds)
    )
    assert max(raw_backgrounds) > min(raw_backgrounds)


def test_repeat_structure_audit_links_mutation_manifest(tmp_path):
    _write_structure(tmp_path / "parent.pdb")
    _write_structure(tmp_path / "repeat.pdb")
    _write_structure(tmp_path / "mutant.pdb", middle="TYR")
    parent = parse_structure(tmp_path / "parent.pdb", "A")
    mutant = parse_structure(tmp_path / "mutant.pdb", "A")
    shifted = mutant.coords.copy()
    shifted[1, :, 2] += 0.3
    pair = pair_parsed_structures(parent, replace(mutant, coords=shifted), "repeat-1")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([PairRecord(pair, "parent", "family", "train")], manifest)
    pairs = tmp_path / "repeats.csv"
    pairs.write_text(
        "pair_id,parent_structure,parent_chain,repeat_structure,repeat_chain,mutation_index\n"
        "repeat-1,parent.pdb,A,repeat.pdb,A,1\n"
    )

    report = audit_repeat_pairs(
        pairs,
        mutation_manifest=manifest,
        min_neighborhood_signal_to_background=2.0,
    )

    assert report["records"][0]["mutant_site_rmsd_angstrom"] > 0
    assert report["records"][0]["site_signal_to_background"] is None
    assert report["pairs"][0]["selected"] is False
    assert report["selected_records"] == 0
    assert np.isfinite(report["summary"]["mutant_site_rmsd_angstrom"]["mean"])


def test_repeat_structure_audit_threshold_requires_manifest(tmp_path):
    _write_structure(tmp_path / "parent.pdb")
    _write_structure(tmp_path / "repeat.pdb")
    pairs = tmp_path / "repeats.csv"
    pairs.write_text(
        "pair_id,parent_structure,parent_chain,repeat_structure,repeat_chain,mutation_index\n"
        "repeat-1,parent.pdb,A,repeat.pdb,A,1\n"
    )

    with pytest.raises(ValueError, match="require mutation_manifest"):
        audit_repeat_pairs(pairs, min_site_signal_to_background=1.0)

    with pytest.raises(ValueError, match="requires mutation_manifest"):
        audit_repeat_pairs(pairs, skip_unlisted_pairs=True)
