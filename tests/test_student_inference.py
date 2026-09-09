import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ospedit.data import StructurePair
from ospedit.student_inference import (
    ParentContextCache,
    apply_student_delta,
    predict_student,
    predict_student_batch,
)
from ospedit.student import SpatialGraphStudent


def make_pair():
    coords = np.array(
        [
            [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.0, 4.8, 0.0]],
        ]
    )
    return StructurePair("student-infer", "AA", "AY", coords, coords.copy(), (1,))


class ZeroStudent(torch.nn.Module):
    def forward(self, parent, edit):
        return torch.zeros(
            (parent.shape[0], parent.shape[1], 6),
            dtype=parent.dtype,
            device=parent.device,
        )


class ShiftStudent(torch.nn.Module):
    def forward(self, parent, edit):
        output = torch.zeros(
            (parent.shape[0], parent.shape[1], 6),
            dtype=parent.dtype,
            device=parent.device,
        )
        output[:, 1, 0] = edit[:, 1, -1]
        return output


class AllShiftStudent(torch.nn.Module):
    def forward(self, parent, edit):
        output = torch.zeros((parent.shape[0], parent.shape[1], 6), dtype=parent.dtype, device=parent.device)
        output[..., 0] = 1.0
        return output


class MaskSpyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, parent, edit, residue_mask=None):
        self.mask = residue_mask.detach().cpu().tolist()
        return torch.zeros(
            (parent.shape[0], parent.shape[1], 6),
            dtype=parent.dtype,
            device=parent.device,
        )


class GeometryMarkerStudent(torch.nn.Module):
    def forward(self, parent, edit, residue_mask=None):
        output = torch.zeros(
            (parent.shape[0], parent.shape[1], 6),
            dtype=parent.dtype,
            device=parent.device,
        )
        output[..., 0] = parent[..., -1]
        return output


class EditSpyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edit = None

    def forward(self, parent, edit, residue_mask=None):
        self.edit = edit.detach().cpu()
        return torch.zeros((parent.shape[0], parent.shape[1], 6), dtype=parent.dtype, device=parent.device)


def test_predict_student_zero_delta_preserves_parent():
    pair = make_pair()
    assert np.array_equal(predict_student(ZeroStudent(), pair), pair.parent_coords)


def test_predict_student_applies_local_translation():
    pair = make_pair()
    prediction = predict_student(ShiftStudent(), pair)
    assert np.allclose(prediction[1, :, 0] - pair.parent_coords[1, :, 0], 1.0)


def test_predict_student_is_global_rigid_equivariant():
    pair = make_pair()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shift = np.array([4.0, -2.0, 1.5])
    transformed = StructurePair(
        "student-equivariant",
        pair.parent_sequence,
        pair.mutant_sequence,
        pair.parent_coords @ rotation.T + shift,
        pair.mutant_coords @ rotation.T + shift,
        pair.mutation_indices,
        pair.atom_names,
    )
    expected = predict_student(ShiftStudent(), pair) @ rotation.T + shift
    actual = predict_student(ShiftStudent(), transformed)
    assert np.allclose(actual, expected, atol=1e-6)


def test_predict_student_masks_degenerate_parent_frames():
    pair = make_pair()
    broken = pair.parent_coords.copy()
    broken[0] = np.nan
    pair = StructurePair(
        "broken",
        pair.parent_sequence,
        pair.mutant_sequence,
        broken,
        pair.mutant_coords,
        pair.mutation_indices,
    )
    model = MaskSpyStudent()
    predict_student(model, pair)
    assert model.mask == [[False, True]]


def test_apply_student_delta_validates_shape():
    with pytest.raises(ValueError, match="shape"):
        apply_student_delta(make_pair(), np.zeros((2, 5)))


def test_student_update_scale_interpolates_from_parent():
    pair = make_pair()
    full = predict_student(ShiftStudent(), pair, update_scale=1.0)
    half = predict_student(ShiftStudent(), pair, update_scale=0.5)
    zero = predict_student(ShiftStudent(), pair, update_scale=0.0)
    assert np.allclose(half - pair.parent_coords, 0.5 * (full - pair.parent_coords))
    assert np.array_equal(zero, pair.parent_coords)
    with pytest.raises(ValueError, match="non-negative"):
        predict_student(ShiftStudent(), pair, update_scale=-1.0)


def test_predict_student_batch_matches_individual_inference():
    pairs = [make_pair(), make_pair()]
    model = ShiftStudent()
    individual = [predict_student(model, pair) for pair in pairs]
    batched = predict_student_batch(model, pairs)
    for expected, actual in zip(individual, batched, strict=True):
        assert np.allclose(expected, actual)


def test_predict_student_batch_handles_mixed_lengths():
    short = make_pair()
    long = make_pair()
    long = StructurePair(
        "long",
        "AAA",
        "AYA",
        np.concatenate((long.parent_coords, long.parent_coords[:1]), axis=0),
        np.concatenate((long.mutant_coords, long.mutant_coords[:1]), axis=0),
        (1,),
    )
    results = predict_student_batch(ShiftStudent(), [short, long])
    assert [result.shape[0] for result in results] == [2, 3]


def test_predict_student_batch_mask_does_not_depend_on_mutant_coordinates():
    pair = make_pair()
    missing_mutant = pair.mutant_coords.copy()
    missing_mutant[0] = np.nan
    pair = StructurePair(
        "missing-target",
        pair.parent_sequence,
        pair.mutant_sequence,
        pair.parent_coords,
        missing_mutant,
        pair.mutation_indices,
        pair.atom_names,
    )
    model = MaskSpyStudent()
    predict_student_batch(model, [pair])
    assert model.mask == [[True, True]]


def test_predict_student_can_ablate_target_identity_but_keep_mutation_position():
    model = EditSpyStudent()
    predict_student(model, make_pair(), include_target_residue=False)
    assert torch.equal(model.edit[..., 20:40], torch.zeros_like(model.edit[..., 20:40]))
    assert model.edit[0, :, 40].tolist() == [0.0, 1.0]


def test_geometry_cached_predictions_match_uncached_and_are_order_independent():
    base = make_pair()
    coords = np.concatenate((base.parent_coords, base.parent_coords[:1] + np.array([0.0, 7.6, 0.0])), axis=0)
    first = StructurePair("first-position", "AAA", "YAA", coords, coords.copy(), (0,))
    second = StructurePair("second-position", "AAA", "AAY", coords.copy(), coords.copy(), (2,))
    model = GeometryMarkerStudent()
    expected = predict_student_batch(model, [first, second], include_geometry=True)

    forward = predict_student_batch(model, [first, second], parent_cache=ParentContextCache(include_geometry=True))
    reverse = predict_student_batch(model, [second, first], parent_cache=ParentContextCache(include_geometry=True))

    assert np.allclose(forward[0], expected[0])
    assert np.allclose(forward[1], expected[1])
    assert np.allclose(reverse[0], expected[1])
    assert np.allclose(reverse[1], expected[0])


def test_spatial_graph_student_runs_single_and_batch_inference():
    pair = make_pair()
    model = SpatialGraphStudent(parent_dim=16, hidden_dim=16, blocks=1)
    single = predict_student(model, pair, include_spatial_graph=True, spatial_neighbors=1)
    batch = predict_student_batch(model, [pair], include_spatial_graph=True, spatial_neighbors=1)
    assert np.allclose(single, batch[0])


def test_biochemical_spatial_graph_student_runs_single_and_batch_inference():
    pair = make_pair()
    model = SpatialGraphStudent(parent_dim=16, edit_dim=48, hidden_dim=16, blocks=1)
    single = predict_student(
        model,
        pair,
        include_spatial_graph=True,
        spatial_neighbors=1,
        include_biochemical=True,
    )
    batch = predict_student_batch(
        model,
        [pair],
        include_spatial_graph=True,
        spatial_neighbors=1,
        include_biochemical=True,
    )
    assert np.allclose(single, batch[0])


def test_output_localization_preserves_remote_residues_in_single_and_batch():
    base = make_pair()
    coords = np.concatenate((base.parent_coords, base.parent_coords[:1] + np.array([0.0, 15.0, 0.0])), axis=0)
    pair = StructurePair("localized-output", "AAA", "AYA", coords, coords.copy(), (1,))
    model = AllShiftStudent()
    single = predict_student(
        model,
        pair,
        output_localization_radius=4.0,
        output_localization_transition=4.0,
    )
    batch = predict_student_batch(
        model,
        [pair],
        output_localization_radius=4.0,
        output_localization_transition=4.0,
    )[0]
    assert not np.array_equal(single[1], pair.parent_coords[1])
    assert np.array_equal(single[2], pair.parent_coords[2])
    assert np.allclose(single, batch)
