import json

import numpy as np

from ospedit.data import PairRecord, StructurePair, manifest_fingerprint, write_manifest
from scripts.build_response_review_queue import build_response_review_queue


def _inputs(tmp_path):
    coords = np.zeros((3, 1, 3))
    pair = StructurePair("pair-1", "AAA", "AYA", coords, coords.copy(), (1,))
    record = PairRecord(pair, "parent-1", "family-1", "dev")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([record], manifest)
    fingerprint = manifest_fingerprint([record])

    endpoint = tmp_path / "endpoint.json"
    endpoint.write_text(json.dumps({
        "format": "ospedit.rcsb_endpoint_environment_audit.v1",
        "manifest_fingerprint": fingerprint,
        "records": [{
            "pair_id": "pair-1",
            "parent_pdb_id": "1ABC",
            "mutant_pdb_id": "2ABC",
            "crystal_form_compatible": True,
            "shared_assembly_signatures": [[1, "homomeric protein", 1]],
            "shared_uniprot": ["P1"],
            "cell_comparison": {"max_relative_length_difference": 0.01},
            "shared_primary_citation_ids": ["doi:10.1000/example"],
            "primary_citation_matched": True,
            "shared_crystal_growth_signatures": ["ph 7.0"],
            "declared_crystal_growth_matched": True,
        }],
    }))
    control = tmp_path / "control.json"
    control.write_text(json.dumps({
        "format": "ospedit.rcsb_environment_audit.v1",
        "manifest_fingerprint": fingerprint,
        "records": [
            {
                "pair_id": "pair-1",
                "repeat_pdb_id": f"{index}XYZ",
                "crystal_form_compatible": True,
                "shared_assembly_signatures": [[1, "homomeric protein", 1]],
                "shared_uniprot": ["P1"],
                "cell_comparison": {"max_relative_length_difference": 0.02},
                "background": {"neighborhood_rmsd_angstrom": 0.2},
            }
            for index in (1, 2)
        ],
    }))
    signal = tmp_path / "signal.json"
    signal.write_text(json.dumps({
        "format": "ospedit.background_signal_diagnostic.v1",
        "manifest_fingerprint": fingerprint,
        "rcsb_crystal_form_records": [{
            "pair_id": "pair-1",
            "local": {"signal_to_background_max": 1.2, "signal_to_background_median": 1.4},
            "site": {"signal_to_background_max": 1.3, "signal_to_background_median": 1.5},
            "distance": {"signal_to_background_max": 1.1, "signal_to_background_median": 1.2},
        }],
    }))
    return manifest, endpoint, control, signal


def test_response_review_queue_requires_multiple_crystal_matched_controls(tmp_path):
    report = build_response_review_queue(*_inputs(tmp_path))

    assert report["summary"] == {"candidates": 1, "splits": {"train": 0, "dev": 1, "test": 0}, "families": 1}
    assert report["records"][0]["mutation"] == [{"index": 1, "source": "A", "target": "Y"}]
    assert report["records"][0]["repeat_control_count"] == 2
    assert report["records"][0]["endpoint_shared_primary_citation_ids"] == [
        "doi:10.1000/example"
    ]
    assert report["records"][0]["endpoint_declared_crystal_growth_matched"] is True
    assert report["usage"] == "manual_review_queue_not_training_or_test_data"


def test_response_review_queue_applies_signal_gate(tmp_path):
    inputs = _inputs(tmp_path)

    report = build_response_review_queue(*inputs, min_local_sbr=1.5)

    assert report["records"] == []
