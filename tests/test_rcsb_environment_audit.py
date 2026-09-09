import json
from pathlib import Path

import numpy as np
import pytest

from ospedit.data import PairRecord, StructurePair
from ospedit.rcsb_metadata import parse_rcsb_entry, pdb_id_from_path
from scripts.audit_rcsb_environment import audit_rcsb_environment
from scripts.audit_rcsb_endpoints import audit_rcsb_endpoints


def _raw_entry(pdb_id: str = "1ABC", protein_count: int = 1) -> dict:
    return {
        "rcsb_id": pdb_id,
        "rcsb_entry_info": {"experimental_method": "X-ray", "resolution_combined": [1.5]},
        "cell": {
            "length_a": 10.0, "length_b": 20.0, "length_c": 30.0,
            "angle_alpha": 90.0, "angle_beta": 90.0, "angle_gamma": 90.0,
        },
        "symmetry": {"space_group_name_H_M": "P 1"},
        "exptl_crystal_grow": [{"method": "VAPOR DIFFUSION", "pH": 7.0, "temp": 277.0}],
        "rcsb_primary_citation": {
            "title": "Study", "year": 2000,
            "pdbx_database_id_DOI": "10.1000/example",
            "pdbx_database_id_PubMed": 12345678,
        },
        "polymer_entities": [{
            "entity_poly": {"rcsb_entity_polymer_type": "Protein", "rcsb_sample_sequence_length": 3},
            "rcsb_polymer_entity": {"pdbx_fragment": None, "pdbx_mutation": None},
            "rcsb_polymer_entity_container_identifiers": {
                "reference_sequence_identifiers": [
                    {"database_name": "UniProt", "database_accession": "P1"}
                ]
            },
            "polymer_entity_instances": [{
                "rcsb_polymer_entity_instance_container_identifiers": {
                    "asym_id": "X", "auth_asym_id": "A", "entity_id": "1"
                }
            }],
        }],
        "assemblies": [{
            "rcsb_assembly_container_identifiers": {"assembly_id": "1"},
            "rcsb_assembly_info": {
                "polymer_composition": "homomeric protein",
                "polymer_entity_instance_count": protein_count,
                "polymer_entity_instance_count_protein": protein_count,
                "nonpolymer_entity_instance_count": 0,
                "total_assembly_buried_surface_area": 0.0,
                "num_interfaces": 0,
            },
            "pdbx_struct_assembly": {
                "details": "author_defined_assembly",
                "method_details": None,
                "oligomeric_count": protein_count,
                "oligomeric_details": "monomeric" if protein_count == 1 else "dimeric",
                "rcsb_details": "author_defined_assembly",
            },
            "pdbx_struct_assembly_gen": [{"asym_id_list": ["X"], "oper_expression": "1"}],
        }],
    }


def _context_report(tmp_path) -> Path:
    path = tmp_path / "context.json"
    path.write_text(json.dumps({
        "format": "ospedit.repeat_control_context_audit.v1",
        "manifest_fingerprint": "fingerprint",
        "records": [{
            "pair_id": "pair-1",
            "parent_id": "parent-1",
            "family_id": "family-1",
            "split": "dev",
            "selected": True,
            "parent": {"path": str(tmp_path / "1ABC_parent.pdb"), "target_chain": "A"},
            "repeat": {"path": str(tmp_path / "2ABC_repeat.pdb"), "target_chain": "A"},
            "background": {"neighborhood_rmsd_angstrom": 0.2},
        }],
    }))
    return path


def test_parse_rcsb_entry_maps_author_to_label_chain():
    parsed = parse_rcsb_entry(_raw_entry())

    assert parsed["chains"]["A"][0]["asym_id"] == "X"
    assert parsed["chains"]["A"][0]["uniprot_accessions"] == ["P1"]
    assert parsed["assemblies"][0]["protein_instance_count"] == 1


def test_pdb_id_from_path_handles_provenance_suffix():
    assert pdb_id_from_path("/tmp/1abc_parent.pdb") == "1ABC"
    with pytest.raises(ValueError, match="cannot infer"):
        pdb_id_from_path("parent.pdb")


def test_rcsb_environment_accepts_shared_assembly_and_crystal_form(tmp_path):
    parent = parse_rcsb_entry(_raw_entry("1ABC"))
    repeat = parse_rcsb_entry(_raw_entry("2ABC"))

    report = audit_rcsb_environment(
        _context_report(tmp_path), {"1ABC": parent, "2ABC": repeat}
    )

    assert report["summary"]["assembly_compatible"]["controls"] == 1
    assert report["summary"]["crystal_form_compatible"]["controls"] == 1
    assert report["records"][0]["shared_uniprot"] == ["P1"]


def test_rcsb_environment_separates_assembly_and_crystal_mismatch(tmp_path):
    parent = parse_rcsb_entry(_raw_entry("1ABC"))
    repeat = parse_rcsb_entry(_raw_entry("2ABC"))
    repeat["space_group"] = "P 2"
    report = audit_rcsb_environment(
        _context_report(tmp_path), {"1ABC": parent, "2ABC": repeat}
    )
    assert report["records"][0]["assembly_compatible"] is True
    assert report["records"][0]["crystal_form_compatible"] is False
    assert report["records"][0]["crystal_rejection_reasons"] == ["space_group_mismatch"]

    repeat = parse_rcsb_entry(_raw_entry("2ABC", protein_count=2))
    report = audit_rcsb_environment(
        _context_report(tmp_path), {"1ABC": parent, "2ABC": repeat}
    )
    assert report["records"][0]["assembly_compatible"] is False
    assert "biological_assembly_signature_mismatch" in report["records"][0]["assembly_rejection_reasons"]


def _endpoint_record(tmp_path) -> PairRecord:
    coords = np.zeros((3, 1, 3))
    pair = StructurePair("pair-1", "AAA", "AYA", coords, coords.copy(), (1,))
    return PairRecord(
        pair,
        "parent-1",
        "family-1",
        "test",
        source_file=str(tmp_path / "1ABC_parent.pdb"),
        target_file=str(tmp_path / "2ABC_mutant.pdb"),
        source_chain="A",
        target_chain="A",
    )


def test_rcsb_endpoint_audit_checks_construct_and_assembly(tmp_path):
    parent = parse_rcsb_entry(_raw_entry("1ABC"))
    mutant = parse_rcsb_entry(_raw_entry("2ABC"))

    report = audit_rcsb_endpoints(
        [_endpoint_record(tmp_path)], {"1ABC": parent, "2ABC": mutant}
    )

    assert report["summary"]["assembly_compatible"]["pairs"] == 1
    assert report["summary"]["crystal_form_compatible"]["pairs"] == 1
    assert report["summary"]["citation_growth_compatible"]["pairs"] == 1

    mutant["chains"]["A"][0]["sample_sequence_length"] = 4
    report = audit_rcsb_endpoints(
        [_endpoint_record(tmp_path)], {"1ABC": parent, "2ABC": mutant}
    )
    assert report["records"][0]["assembly_compatible"] is False
    assert "deposited_construct_length_mismatch" in report["records"][0]["assembly_rejection_reasons"]
