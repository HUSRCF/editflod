import numpy as np
import pytest

from ospedit.data import StructurePair
from ospedit.metrics import evaluate_pair
from ospedit.oracle import oracle_local_delta, oracle_prediction


def make_pair() -> StructurePair:
    residue = np.array(
        [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    )
    parent = np.repeat(residue[None], 2, axis=0)
    parent[1, :, 1] += 4.0
    mutant = parent.copy()
    mutant[1, :, 0] += 0.4
    return StructurePair("oracle", "AA", "AY", parent, mutant, (1,))


def test_unbounded_oracle_reconstructs_rigid_residue_target():
    pair = make_pair()
    prediction = oracle_prediction(pair)
    assert evaluate_pair(pair, prediction)["local_backbone_error"] == pytest.approx(0.0, abs=1e-6)


def test_bounded_oracle_matches_student_component_bound():
    pair = make_pair()
    delta, valid = oracle_local_delta(pair, max_normalized_delta=0.1)
    assert valid.all()
    assert np.max(np.abs(delta)) == pytest.approx(0.1)
    assert evaluate_pair(pair, oracle_prediction(pair, max_normalized_delta=0.1))["local_backbone_error"] > 0


def test_oracle_rejects_nonpositive_bound():
    with pytest.raises(ValueError, match="positive"):
        oracle_local_delta(make_pair(), max_normalized_delta=0.0)


def test_localized_oracle_preserves_remote_parent_coordinates():
    pair = make_pair()
    prediction = oracle_prediction(pair, localization_radius=1.0, localization_transition=1.0)
    assert np.array_equal(prediction[0], pair.parent_coords[0])
    assert np.allclose(prediction[1], pair.mutant_coords[1], atol=1e-6)
