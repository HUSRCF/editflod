import numpy as np

from ospedit.data import StructurePair
from ospedit.diagnostics import (
    assess_teacher_admission,
    condition_response_diagnostic,
    condition_response_repeat_error,
)
from ospedit.geometry import residue_frames, so3_exp


class DiagnosticEndpoint:
    def endpoint(self, coords, sequence, noise_level):
        rotations, origins, _ = residue_frames(coords, ("N", "CA", "C", "O"))
        rotations = rotations.copy()
        origins = origins.copy()
        if sequence[1] == "Y":
            origins[1] += rotations[1] @ np.array([noise_level + 0.5, 0.0, 0.0])
            rotations[1] = rotations[1] @ so3_exp(np.array([0.0, 0.0, 0.1]))
        return rotations, origins


def test_condition_response_reports_mutation_signal():
    coords = np.array(
        [
            [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.0, 4.8, 0.0]],
        ]
    )
    pair = StructurePair("diag", "AA", "AY", coords, coords.copy(), (1,))
    report = condition_response_diagnostic(DiagnosticEndpoint(), pair, [0.0, 0.5])
    assert report["levels"][0]["mutation_translation_response"] == 0.5
    assert report["levels"][0]["remote_translation_response"] == 0.0
    assert report["levels"][0]["mutation_rotation_response"] > 0


def test_condition_response_is_zero_for_identical_sequences():
    coords = np.array(
        [
            [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.0, 4.8, 0.0]],
        ]
    )
    pair = StructurePair("identity", "AA", "AA", coords, coords.copy(), ())
    level = condition_response_diagnostic(DiagnosticEndpoint(), pair, [0.0])["levels"][0]
    assert level["mean_translation_response"] == 0.0
    assert level["mean_rotation_response"] == 0.0


def test_teacher_admission_accepts_finite_mutation_signal():
    coords = np.array(
        [
            [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.0, 4.8, 0.0]],
        ]
    )
    pair = StructurePair("admit", "AA", "AY", coords, coords.copy(), (1,))
    report = condition_response_diagnostic(DiagnosticEndpoint(), pair, [0.0])
    decision = assess_teacher_admission(report, min_mutation_response=0.1)
    assert decision["accepted"] is True


def test_teacher_admission_rejects_no_signal():
    coords = np.array(
        [
            [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.0, 4.8, 0.0]],
        ]
    )
    pair = StructurePair("reject", "AA", "AY", coords, coords.copy(), (1,))

    class ZeroEndpoint:
        def endpoint(self, coords, sequence, noise_level):
            rotations, origins, _ = residue_frames(coords, ("N", "CA", "C", "O"))
            return rotations, origins

    report = condition_response_diagnostic(ZeroEndpoint(), pair, [0.0])
    decision = assess_teacher_admission(report)
    assert decision["accepted"] is False
    assert decision["reasons"]


def test_condition_response_repeat_error_is_zero_for_matching_reports():
    report = {"levels": [{"mean_translation_response": 1.0, "mutation_translation_response": 2.0,
                           "remote_translation_response": 0.5, "mean_rotation_response": 0.1,
                           "mutation_rotation_response": 0.2, "remote_rotation_response": 0.0}]}
    assert condition_response_repeat_error(report, report) == {
        "max_translation_error": 0.0,
        "max_rotation_error": 0.0,
    }


def test_teacher_admission_rejects_unstable_response():
    report = {"levels": [{"mutation_translation_response": 1.0, "mutation_rotation_response": 0.0}]}
    decision = assess_teacher_admission(
        report,
        repeat_error={"max_translation_error": 1e-3, "max_rotation_error": 0.0},
        max_repeat_error=1e-4,
    )
    assert decision["accepted"] is False
    assert "repeat error" in " ".join(decision["reasons"])
