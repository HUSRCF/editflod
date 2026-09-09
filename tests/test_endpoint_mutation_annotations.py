import numpy as np

from ospedit.data import PairRecord, StructurePair
from ospedit.pdb_annotations import parse_engineered_mutations
from scripts.audit_endpoint_mutation_annotations import audit_endpoint_mutation_annotations


def _pdb(path, seqadv=""):
    path.write_text(seqadv + "\nEND\n")
    return str(path)


def _record(tmp_path, target_seqadv=""):
    coords = np.zeros((3, 1, 3))
    pair = StructurePair("pair-1", "AAA", "AYA", coords, coords.copy(), (1,))
    return PairRecord(
        pair,
        "parent-1",
        "family-1",
        "train",
        source_file=_pdb(tmp_path / "parent.pdb"),
        target_file=_pdb(tmp_path / "mutant.pdb", target_seqadv),
        source_chain="A",
        target_chain="A",
        residue_map=(("A", 9, ""), ("A", 10, ""), ("A", 11, "")),
    )


def test_parse_engineered_mutation_seqadv(tmp_path):
    path = tmp_path / "mutant.pdb"
    _pdb(path, "SEQADV 1ABC TYR A   10  UNP  P00000001 ALA    10 ENGINEERED MUTATION")

    assert parse_engineered_mutations(path) == [{
        "chain": "A",
        "residue_number": 10,
        "insertion_code": "",
        "deposited_residue_name": "TYR",
        "deposited_residue": "Y",
        "reference_residue_name": "ALA",
        "reference_residue": "A",
    }]


def test_annotation_audit_supports_manifest_edit(tmp_path):
    seqadv = "SEQADV 1ABC TYR A   10  UNP  P00000001 ALA    10 ENGINEERED MUTATION"
    report = audit_endpoint_mutation_annotations([_record(tmp_path, seqadv)])

    assert report["summary"]["classifications"] == {"edit_supported_by_seqadv": 1}
    assert report["records"][0]["direct_support"] is True


def test_annotation_audit_flags_other_engineered_site(tmp_path):
    seqadv = "SEQADV 1ABC ASP A   19  UNP  P00000001 ASN    19 ENGINEERED MUTATION"
    report = audit_endpoint_mutation_annotations([_record(tmp_path, seqadv)])

    assert report["records"][0]["classification"] == "other_engineered_mutation_only"


def test_annotation_audit_supports_two_variants_at_same_site(tmp_path):
    parent = "SEQADV 1ABC ALA A   10  UNP  P00000001 GLN    10 ENGINEERED MUTATION"
    mutant = "SEQADV 2ABC TYR A   10  UNP  P00000001 GLN    10 ENGINEERED MUTATION"
    record = _record(tmp_path, mutant)
    _pdb(tmp_path / "parent.pdb", parent)

    report = audit_endpoint_mutation_annotations([record])

    assert report["records"][0]["paired_endpoint_support"] is True
    assert report["records"][0]["classification"] == "edit_supported_by_seqadv"
