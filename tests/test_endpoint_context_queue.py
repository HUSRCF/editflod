import json

import numpy as np
import pytest

from ospedit.data import PairRecord, StructurePair, manifest_fingerprint, write_manifest
from scripts.build_endpoint_context_queue import build_endpoint_context_queue


def _inputs(tmp_path):
    coords = np.zeros((3, 1, 3))
    pair = StructurePair("pair-1", "AAA", "AYA", coords, coords.copy(), (1,))
    record = PairRecord(pair, "parent-1", "family-1", "test")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([record], manifest)
    endpoint = tmp_path / "endpoint.json"
    endpoint.write_text(json.dumps({
        "format": "ospedit.rcsb_endpoint_environment_audit.v1",
        "manifest_fingerprint": manifest_fingerprint([record]),
        "metadata": {
            "1ABC": {"title": "Parent", "primary_citation": {"year": 2000}},
            "2ABC": {"title": "Mutant", "primary_citation": {"year": 2000}},
        },
        "records": [{
            "pair_id": "pair-1",
            "parent_id": "parent-1",
            "family_id": "family-1",
            "split": "test",
            "parent_pdb_id": "1ABC",
            "mutant_pdb_id": "2ABC",
            "citation_crystal_compatible": True,
            "citation_growth_compatible": False,
            "shared_primary_citation_ids": ["PMID:1"],
            "shared_assembly_signatures": [[1, "homomeric protein", 1]],
            "shared_uniprot": ["P1"],
            "cell_comparison": {"max_relative_length_difference": 0.01},
            "shared_crystal_growth_signatures": [],
            "declared_crystal_growth_matched": False,
        }],
    }))
    return manifest, endpoint


def test_endpoint_context_queue_does_not_require_response_report(tmp_path):
    report = build_endpoint_context_queue(*_inputs(tmp_path))

    assert report["summary"] == {
        "candidates": 1,
        "families": 1,
        "declared_growth_matched": 0,
        "splits": {"train": 0, "dev": 0, "test": 1},
    }
    assert report["configuration"]["observed_structure_response_used_for_selection"] is False
    assert report["records"][0]["frozen_test_record"] is True
    assert report["records"][0]["mutation"] == [
        {"index": 1, "source": "A", "target": "Y"}
    ]


def test_endpoint_context_queue_supports_stricter_growth_tier(tmp_path):
    report = build_endpoint_context_queue(
        *_inputs(tmp_path), tier="citation_growth_compatible"
    )
    assert report["records"] == []


def test_endpoint_context_queue_rejects_unknown_tier(tmp_path):
    with pytest.raises(ValueError, match="tier must be"):
        build_endpoint_context_queue(*_inputs(tmp_path), tier="response_size")


def test_endpoint_context_queue_attaches_annotation_without_filtering(tmp_path):
    manifest, endpoint = _inputs(tmp_path)
    annotation = tmp_path / "annotations.json"
    annotation.write_text(json.dumps({
        "format": "ospedit.endpoint_mutation_annotation_audit.v1",
        "manifest_fingerprint": json.loads(endpoint.read_text())["manifest_fingerprint"],
        "records": [{
            "pair_id": "pair-1",
            "classification": "other_engineered_mutation_only",
            "parent_engineered_mutations": [],
            "mutant_engineered_mutations": [{"residue_number": 99}],
        }],
    }))

    report = build_endpoint_context_queue(
        manifest, endpoint, annotation_report=annotation
    )

    assert report["summary"]["candidates"] == 1
    assert report["records"][0]["mutation_annotation_classification"] == (
        "other_engineered_mutation_only"
    )
    assert report["summary"]["mutation_annotation_classifications"] == {
        "other_engineered_mutation_only": 1
    }
    assert report["configuration"]["mutation_annotation_used_for_selection"] is False
