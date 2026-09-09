import numpy as np
import pytest

from ospedit.data import StructurePair
from ospedit.metrics import backbone_angle_violations, backbone_clash_violations, backbone_geometry_violations, evaluate_pair, region_masks
from ospedit.models import ConditionalDifferenceEditor, CopyParentEditor, TargetUpdateEditor


class SequenceField:
    def field(self, coords, sequence, noise_level):
        out = np.zeros_like(coords)
        if sequence[1] == "Y":
            out[1, :, 0] = 1.0
        return out


def pair():
    parent = np.zeros((3, 1, 3))
    mutant = parent.copy()
    mutant[1, :, 0] = 1.0
    return StructurePair("toy", "AAA", "AYA", parent, mutant, (1,))


def test_copy_parent_has_zero_predicted_change():
    result = evaluate_pair(pair(), CopyParentEditor().predict(pair()))
    assert result["distance_change_error"] > 0
    assert result["local_distance_change_error"] >= 0
    assert result["mutation_site_backbone_error"] > 0.0
    assert result["mutation_site_global_rmsd"] > 0.0


def test_conditional_difference_applies_sequence_change():
    result = evaluate_pair(pair(), ConditionalDifferenceEditor(SequenceField(), step_size=1).predict(pair()))
    assert result["distance_change_error"] == 0


def test_distance_change_error_excludes_diagonal_pairs():
    source = pair()
    prediction = source.parent_coords.copy()
    metrics = evaluate_pair(source, prediction)
    parent = source.parent_coords[:, 0]
    target = source.mutant_coords[:, 0]
    true_delta = np.linalg.norm(target[:, None] - target[None, :], axis=-1) - np.linalg.norm(parent[:, None] - parent[None, :], axis=-1)
    off_diagonal = ~np.eye(len(parent), dtype=bool)
    expected = np.sqrt(np.mean(true_delta[off_diagonal] ** 2))
    assert np.isclose(metrics["distance_change_error"], expected)


def test_distance_change_reports_signed_response_alignment():
    source = pair()
    exact = evaluate_pair(source, source.mutant_coords)
    copied = evaluate_pair(source, source.parent_coords)
    assert np.isclose(exact["distance_change_cosine"], 1.0)
    assert exact["predicted_distance_change_norm"] > 0
    assert copied["predicted_distance_change_norm"] == 0.0
    assert np.isnan(copied["distance_change_cosine"])
    assert np.isclose(exact["local_distance_change_cosine"], 1.0)
    assert np.isclose(exact["mutation_site_backbone_error"], 0.0)
    assert np.isclose(exact["mutation_site_global_rmsd"], 0.0)


def test_conditional_difference_is_exact_identity_without_mutation():
    source = pair()
    identity = StructurePair("identity", "AAA", "AAA", source.parent_coords, source.parent_coords.copy(), ())
    result = ConditionalDifferenceEditor(SequenceField(), source_weight=3.0).predict(identity)
    assert np.array_equal(result, identity.parent_coords)


def test_conditional_difference_shares_explicit_noise_state():
    class StochasticField:
        def __init__(self):
            self.states = []

        def field(self, coords, sequence, noise_level, *, noise_state):
            self.states.append(noise_state)
            return np.zeros_like(coords)

    model = StochasticField()
    ConditionalDifferenceEditor(model, noise_level=0.4).predict(pair())
    assert len(model.states) == 2
    assert model.states[0] is model.states[1]
    assert model.states[0].level == 0.4


def test_target_update_is_exact_identity_without_mutation():
    source = pair()
    identity = StructurePair("identity-target", "AAA", "AAA", source.parent_coords, source.parent_coords.copy(), ())
    result = TargetUpdateEditor(SequenceField()).predict(identity)
    assert np.array_equal(result, identity.parent_coords)


def test_structure_pair_rejects_noninteger_duplicate_and_out_of_bounds_indices():
    coords = np.zeros((3, 1, 3))
    for indices in ((1.0,), (1, 1), (3,)):
        try:
            StructurePair("bad-index", "AAA", "AYA", coords, coords.copy(), indices)
        except ValueError as error:
            assert "mutation_indices" in str(error)
        else:
            raise AssertionError("expected mutation index validation error")


def test_regions_use_named_ca_atom_in_n_ca_c_o_layout():
    parent = np.zeros((2, 4, 3))
    parent[1, 0, 0] = 100.0  # N is far away, CA remains nearby.
    mutant = parent.copy()
    structure = StructurePair("atoms", "AA", "AY", parent, mutant, (1,))
    assert structure.ca_atom_index == 1
    assert region_masks(structure)["local"].tolist() == [True, True]


def test_backbone_geometry_violations_respect_missing_atoms():
    good = np.array([[[0.0, 0.0, 0.0], [1.46, 0.0, 0.0], [2.99, 0.0, 0.0], [4.23, 0.0, 0.0]]])
    bad = good.copy()
    bad[0, 1, 0] = 2.0
    assert backbone_geometry_violations(good, ("N", "CA", "C", "O")) == (0, 3)
    assert backbone_geometry_violations(bad, ("N", "CA", "C", "O"))[0] == 2
    missing = good.copy()
    missing[0, 3] = np.nan
    assert backbone_geometry_violations(missing, ("N", "CA", "C", "O")) == (0, 2)


def test_evaluate_pair_reports_coordinate_frame_drift():
    coords = np.array(
        [[[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
         [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.0, 4.8, 0.0]]]
    )
    pair = StructurePair("frame", "AA", "AY", coords, coords.copy(), (1,))
    translated = coords + np.array([10.0, 0.0, 0.0])
    metrics = evaluate_pair(pair, translated)
    assert metrics["parent_to_prediction_frame_rmsd"] > 1.0
    assert metrics["parent_to_prediction_rmsd"] == pytest.approx(0.0, abs=1e-6)


def test_backbone_angle_violations_count_valid_angles():
    coords = np.array([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.46, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    violations, checked = backbone_angle_violations(coords, ("N", "CA", "C", "O"))
    assert (violations, checked) == (2, 2)


def test_backbone_clashes_ignore_adjacent_residues():
    residue = np.array([[0.0, 0.0, 0.0], [1.46, 0.0, 0.0], [2.99, 0.0, 0.0], [4.2, 0.0, 0.0]])
    coords = np.stack([residue, residue + [0.0, 3.8, 0.0], residue + [0.0, 0.5, 0.0]])
    violations, checked = backbone_clash_violations(coords, ("N", "CA", "C", "O"))
    assert checked == 16
    assert violations > 0
