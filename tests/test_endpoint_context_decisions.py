from copy import deepcopy

from scripts.validate_endpoint_context_decisions import (
    _payload_sha256,
    validate_endpoint_context_decisions,
)


def _payloads():
    queue = {
        "format": "ospedit.endpoint_context_review_queue.v1",
        "manifest_fingerprint": "fingerprint",
        "records": [
            {"pair_id": "candidate", "split": "train"},
            {"pair_id": "alias", "split": "train"},
            {"pair_id": "frozen", "split": "test"},
        ],
    }
    decisions = {
        "format": "ospedit.endpoint_context_decisions.v1",
        "manifest_fingerprint": "fingerprint",
        "selection_used_observed_response": False,
        "summary": {
            "reviewed": 2,
            "assembly_dependent_challenge": 0,
            "matched_context_candidate": 1,
            "provenance_reject": 0,
            "reverse_duplicate_alias": 1,
            "state_mismatch_challenge": 0,
            "direct_supervision_candidates": {"train": 1, "dev": 0, "test": 0},
            "strict_mutation_attribution_admitted": 0,
        },
        "records": [
            {
                "pair_id": "candidate",
                "split": "train",
                "disposition": "matched_context_candidate",
                "direct_supervision_candidate": True,
            },
            {
                "pair_id": "alias",
                "split": "train",
                "disposition": "reverse_duplicate_alias",
                "direct_supervision_candidate": False,
                "canonical_pair_id": "candidate",
            },
        ],
    }
    decisions["source_queue_sha256"] = _payload_sha256(queue)
    return queue, decisions


def test_endpoint_context_decisions_validate_complete_non_test_review():
    assert validate_endpoint_context_decisions(*_payloads()) == []


def test_endpoint_context_decisions_reject_test_and_summary_drift():
    queue, decisions = _payloads()
    broken = deepcopy(decisions)
    broken["records"].append({
        "pair_id": "frozen",
        "split": "test",
        "disposition": "matched_context_candidate",
        "direct_supervision_candidate": True,
    })

    errors = validate_endpoint_context_decisions(queue, broken)

    assert any("frozen test record" in error for error in errors)
    assert any("unexpected decisions" in error for error in errors)
    assert any("summary does not match" in error for error in errors)
