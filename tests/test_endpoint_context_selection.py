from copy import deepcopy

import numpy as np
import pytest

from ospedit.data import PairRecord, StructurePair, manifest_fingerprint
from scripts.select_endpoint_context_candidates import (
    select_endpoint_context_candidates,
)


def _record(pair_id: str, split: str) -> PairRecord:
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
    return PairRecord(pair, f"parent-{pair_id}", f"family-{pair_id}", split)


def _decisions(records: list[PairRecord]) -> dict:
    return {
        "format": "ospedit.endpoint_context_decisions.v1",
        "manifest_fingerprint": manifest_fingerprint(records),
        "selection_used_observed_response": False,
        "records": [
            {
                "pair_id": "train",
                "split": "train",
                "disposition": "matched_context_candidate",
                "direct_supervision_candidate": True,
            },
            {
                "pair_id": "dev",
                "split": "dev",
                "disposition": "matched_context_candidate",
                "direct_supervision_candidate": True,
            },
            {
                "pair_id": "test",
                "split": "test",
                "disposition": "state_mismatch_challenge",
                "direct_supervision_candidate": False,
            },
        ],
    }


def test_select_endpoint_context_candidates_excludes_frozen_test():
    records = [_record("train", "train"), _record("dev", "dev"), _record("test", "test")]

    selected, report = select_endpoint_context_candidates(records, _decisions(records))

    assert [record.pair.pair_id for record in selected] == ["dev", "train"]
    assert report["summary"]["splits"] == {"train": 1, "dev": 1, "test": 0}
    assert report["selection_used_observed_response"] is False
    assert len(report["decision_sha256"]) == 64


def test_select_endpoint_context_candidates_rejects_test_selection():
    records = [_record("train", "train"), _record("dev", "dev"), _record("test", "test")]
    decisions = deepcopy(_decisions(records))
    decisions["records"][-1].update({
        "disposition": "matched_context_candidate",
        "direct_supervision_candidate": True,
    })

    with pytest.raises(ValueError, match="frozen test"):
        select_endpoint_context_candidates(records, decisions)


def test_select_endpoint_context_candidates_rejects_manifest_drift():
    records = [_record("train", "train"), _record("dev", "dev"), _record("test", "test")]
    decisions = _decisions(records)
    changed = [*records, _record("extra", "train")]

    with pytest.raises(ValueError, match="different manifest"):
        select_endpoint_context_candidates(changed, decisions)
