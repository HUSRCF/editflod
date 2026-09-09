import io
import json
from pathlib import Path

from ospedit.data import pair_record_from_structures, parse_structure
import scripts.discover_rcsb_repeats as repeat_discovery
from scripts.discover_rcsb_repeats import (
    discover_repeat_rows,
    download_rcsb_entries,
    rcsb_sequence_entries,
)


def _pdb(path: Path, residues: tuple[str, ...], hetero: tuple[str, ...]) -> None:
    lines = ["HEADER    TEST", "EXPDTA    X-RAY DIFFRACTION"]
    serial = 1
    for number, residue in enumerate(residues, start=1):
        for atom, x in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0)):
            lines.append(
                f"ATOM  {serial:5d} {atom:>4s} {residue:>3s} A{number:4d}"
                f"    {x + number:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
            )
            serial += 1
    for index, residue in enumerate(hetero, start=900):
        lines.append(
            f"HETATM{serial:5d}  C1  {residue:>3s} A{index:4d}"
            f"    {float(index):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
        )
        serial += 1
    path.write_text("\n".join(lines) + "\nEND\n")


def test_repeat_discovery_filters_observed_sequence_and_context(tmp_path):
    parent_path = tmp_path / "1AAA.pdb"
    mutant_path = tmp_path / "1AAB.pdb"
    repeat_path = tmp_path / "2AAA.pdb"
    wrong_ligand_path = tmp_path / "3AAA.pdb"
    wrong_sequence_path = tmp_path / "4AAA.pdb"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    duplicate_parent_path = cache_dir / "1AAA.pdb"
    _pdb(parent_path, ("ALA", "GLY"), ("ATP",))
    _pdb(mutant_path, ("VAL", "GLY"), ("ATP",))
    _pdb(repeat_path, ("ALA", "GLY"), ("ATP", "CL"))
    _pdb(wrong_ligand_path, ("ALA", "GLY"), ("ADP",))
    _pdb(wrong_sequence_path, ("SER", "GLY"), ("ATP",))
    _pdb(duplicate_parent_path, ("ALA", "GLY"), ("ATP",))
    record = pair_record_from_structures(
        parse_structure(parent_path, "A"),
        parse_structure(mutant_path, "A"),
        pair_id="pair-1",
        parent_id="parent-1",
        family_id="family-1",
        split="dev",
    )

    rows, counters = discover_repeat_rows(
        [record],
        [
            parent_path,
            repeat_path,
            wrong_ligand_path,
            wrong_sequence_path,
            duplicate_parent_path,
        ],
        ignored_hetero=("HOH", "CL"),
    )

    assert [(row["repeat_structure"], row["repeat_chain"]) for row in rows] == [
        (str(repeat_path.resolve()), "A")
    ]
    assert counters["repeat_rows"] == 1
    assert counters["pairs_with_repeats"] == 1
    assert counters["target_hetero_mismatch"] == 1


def test_repeat_download_uses_complete_cached_files_in_sorted_order(tmp_path):
    (tmp_path / "2BBB.pdb").write_text("cached-b")
    (tmp_path / "1AAA.cif").write_text("cached-a")

    paths, failures = download_rcsb_entries(
        ["2BBB", "1AAA", "2BBB"], tmp_path, max_workers=2
    )

    assert [path.name for path in paths] == ["1AAA.cif", "2BBB.pdb"]
    assert failures == []


def test_sequence_search_retries_an_empty_response(monkeypatch):
    responses = iter([
        b"",
        json.dumps({"result_set": [{"identifier": "1ABC_1"}]}).encode(),
    ])

    monkeypatch.setattr(repeat_discovery, "urlopen", lambda request, timeout: io.BytesIO(next(responses)))
    monkeypatch.setattr(repeat_discovery.time, "sleep", lambda _: None)

    assert rcsb_sequence_entries("AG", retries=1) == ["1ABC"]
