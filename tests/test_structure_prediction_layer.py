import numpy as np
import pytest

from ospedit.data import PairRecord, StructurePair, manifest_fingerprint
from scripts.select_structure_prediction_layer import select_structure_prediction_layer


def _record(pair_id: str, split: str, family: str | None = None) -> PairRecord:
    coords = np.zeros((3, 4, 3))
    pair = StructurePair(
        pair_id,
        "AAA",
        "AYA",
        coords,
        coords.copy(),
        (1,),
        ("N", "CA", "C", "O"),
    )
    return PairRecord(pair, f"parent-{pair_id}", family or f"family-{pair_id}", split)


def _reports(records: list[PairRecord]):
    fingerprint = manifest_fingerprint(records)
    endpoint = {
        "format": "ospedit.rcsb_endpoint_environment_audit.v1",
        "manifest_fingerprint": fingerprint,
        "records": [
            {"pair_id": record.pair.pair_id, "crystal_form_compatible": True}
            for record in records
        ],
    }
    context = {
        "format": "ospedit.structure_context_audit.v1",
        "manifest_fingerprint": fingerprint,
        "records": [
            {"pair_id": record.pair.pair_id, "selected": True, "rejection_reasons": []}
            for record in records
        ],
    }
    annotations = {
        "format": "ospedit.endpoint_mutation_annotation_audit.v1",
        "manifest_fingerprint": fingerprint,
        "records": [
            {"pair_id": record.pair.pair_id, "classification": "edit_supported_by_seqadv"}
            for record in records
        ],
    }
    decisions = {
        "format": "ospedit.endpoint_context_decisions.v1",
        "manifest_fingerprint": fingerprint,
        "selection_used_observed_response": False,
        "records": [],
    }
    return endpoint, context, annotations, decisions


def test_structure_prediction_layer_combines_gates_without_test_records():
    records = [
        _record("train", "train"),
        _record("dev", "dev"),
        _record("test", "test"),
        _record("manual-reject", "train"),
    ]
    endpoint, context, annotations, decisions = _reports(records)
    decisions["records"].append({
        "pair_id": "manual-reject",
        "disposition": "state_mismatch_challenge",
    })

    selected, report = select_structure_prediction_layer(
        records, endpoint, context, annotations, decisions
    )

    assert {record.pair.pair_id for record in selected} == {"train", "dev"}
    assert report["frozen_test_records_processed"] == 0
    assert report["summary"]["records_considered"] == 3
    assert report["summary"]["selected_splits"] == {
        "dev": {"records": 1, "families": 1},
        "train": {"records": 1, "families": 1},
    }
    rejected = next(
        row for row in report["records"] if row["pair_id"] == "manual-reject"
    )
    assert rejected["rejection_reasons"] == [
        "manual_context:state_mismatch_challenge"
    ]


def test_structure_prediction_layer_reports_automatic_rejections():
    records = [_record("candidate", "train"), _record("bad", "train")]
    endpoint, context, annotations, decisions = _reports(records)
    endpoint["records"][1]["crystal_form_compatible"] = False
    context["records"][1].update({
        "selected": False,
        "rejection_reasons": ["proximal_hetero_mismatch"],
    })
    annotations["records"][1]["classification"] = "other_engineered_mutation_only"

    selected, report = select_structure_prediction_layer(
        records, endpoint, context, annotations, decisions
    )

    assert [record.pair.pair_id for record in selected] == ["candidate"]
    assert report["summary"]["rejection_reasons"] == {
        "conflicting_mutation_provenance": 1,
        "crystal_form_incompatible": 1,
        "target_context:proximal_hetero_mismatch": 1,
    }


def test_structure_prediction_layer_rejects_report_manifest_drift():
    records = [_record("candidate", "train")]
    endpoint, context, annotations, decisions = _reports(records)
    context["manifest_fingerprint"] = "different"

    with pytest.raises(ValueError, match="different manifest"):
        select_structure_prediction_layer(
            records, endpoint, context, annotations, decisions
        )
