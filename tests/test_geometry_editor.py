import numpy as np

from ospedit.data import StructurePair
from ospedit.geometry import residue_frames, so3_exp, so3_log
from ospedit.models import LocalFrameDifferenceEditor, MultiNoiseLocalFrameDifferenceEditor, MutationNeighborhoodDifferenceEditor


def backbone_pair(parent_sequence="AA", mutant_sequence="AY"):
    coords = np.array(
        [
            [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.0, 4.8, 0.0]],
        ],
        dtype=float,
    )
    return StructurePair("frames", parent_sequence, mutant_sequence, coords, coords.copy(), (1,) if parent_sequence != mutant_sequence else ())


class SequenceEndpoint:
    def endpoint(self, coords, sequence, noise_level):
        rotations, origins, _ = residue_frames(coords, ("N", "CA", "C", "O"))
        if sequence[1] == "Y":
            origins = origins.copy()
            origins[1] += rotations[1] @ np.array([noise_level + 0.5, 0.0, 0.0])
        return rotations, origins


def test_so3_exp_log_roundtrip():
    vector = np.array([0.2, -0.1, 0.3])
    assert np.allclose(so3_log(so3_exp(vector)), vector, atol=1e-7)


def test_so3_log_is_stable_at_pi():
    rotation = so3_exp(np.array([np.pi, 0.0, 0.0]))
    recovered = so3_log(rotation)
    assert np.isfinite(recovered).all()
    assert np.allclose(so3_exp(recovered), rotation, atol=1e-6)


def test_so3_log_is_stable_near_pi_for_arbitrary_axis():
    vector = (np.pi - 1e-7) * np.array([0.3, -0.4, 0.5]) / np.linalg.norm([0.3, -0.4, 0.5])
    recovered = so3_log(so3_exp(vector))
    assert np.allclose(so3_exp(recovered), so3_exp(vector), atol=1e-6)


def test_local_frame_editor_applies_target_difference():
    pair = backbone_pair()
    prediction = LocalFrameDifferenceEditor(SequenceEndpoint()).predict(pair)
    assert np.allclose(prediction[0], pair.parent_coords[0])
    assert np.allclose(prediction[1, 1] - pair.parent_coords[1, 1], [0.5, 0.0, 0.0])


def test_local_frame_editor_is_exact_identity_without_mutation():
    pair = backbone_pair("AA", "AA")
    prediction = LocalFrameDifferenceEditor(SequenceEndpoint()).predict(pair)
    assert np.array_equal(prediction, pair.parent_coords)


def test_mutation_neighborhood_editor_gates_remote_residues():
    pair = backbone_pair()
    prediction = MutationNeighborhoodDifferenceEditor(SequenceEndpoint(), radius=0.0).predict(pair)
    assert np.array_equal(prediction[0], pair.parent_coords[0])
    assert np.allclose(prediction[1, 1] - pair.parent_coords[1, 1], [0.5, 0.0, 0.0])


def test_multi_noise_editor_combines_weighted_responses():
    pair = backbone_pair()
    single = LocalFrameDifferenceEditor(SequenceEndpoint(), noise_level=0.25).predict(pair)
    multi = MultiNoiseLocalFrameDifferenceEditor(SequenceEndpoint(), (0.0, 0.5), (0.25, 0.75)).predict(pair)
    # The fake endpoint's translation is noise + 0.5, so the weighted response is 0.875.
    assert np.allclose(multi[1, 1] - pair.parent_coords[1, 1], [0.875, 0.0, 0.0])
    assert not np.allclose(single[1, 1], multi[1, 1])


def test_multi_noise_editor_requires_matching_weights():
    try:
        MultiNoiseLocalFrameDifferenceEditor(SequenceEndpoint(), (0.1, 0.2), (1.0,))
    except ValueError as error:
        assert "equal length" in str(error)
    else:
        raise AssertionError("expected a weight length validation error")


def test_multi_noise_editor_shares_noise_state_per_level():
    pair = backbone_pair()

    class SharedEndpoint:
        def __init__(self):
            self.calls = []

        def endpoint(self, coords, sequence, noise_level, *, noise_state):
            self.calls.append((sequence, noise_state))
            rotations, origins, _ = residue_frames(coords, ("N", "CA", "C", "O"))
            return rotations, origins

    model = SharedEndpoint()
    MultiNoiseLocalFrameDifferenceEditor(model, (0.2, 0.8)).predict(pair)
    assert model.calls[0][1] is model.calls[1][1]
    assert model.calls[2][1] is model.calls[3][1]
    assert model.calls[0][1] is not model.calls[2][1]


def test_independent_noise_editor_uses_distinct_states():
    from ospedit.models import IndependentNoiseLocalFrameDifferenceEditor

    pair = backbone_pair()

    class IndependentEndpoint:
        def __init__(self):
            self.states = []

        def endpoint(self, coords, sequence, noise_level, *, noise_state):
            self.states.append(noise_state)
            rotations, origins, _ = residue_frames(coords, ("N", "CA", "C", "O"))
            return rotations, origins

    model = IndependentEndpoint()
    IndependentNoiseLocalFrameDifferenceEditor(model, noise_level=0.2).predict(pair)
    assert len(model.states) == 2
    assert model.states[0] is not model.states[1]


def test_repeated_single_noise_editor_uses_matched_four_query_budget():
    from ospedit.models import RepeatedSingleNoiseLocalFrameDifferenceEditor
    pair = backbone_pair()

    class Endpoint:
        def __init__(self):
            self.states = []

        def endpoint(self, coords, sequence, noise_level, *, noise_state):
            self.states.append(noise_state)
            rotations, origins, _ = residue_frames(coords, ("N", "CA", "C", "O"))
            return rotations, origins

    model = Endpoint()
    RepeatedSingleNoiseLocalFrameDifferenceEditor(model, noise_level=0.2, repeats=2).predict(pair)
    assert len(model.states) == 4
    assert model.states[0] is model.states[1]
    assert model.states[2] is model.states[3]
    assert model.states[0] is not model.states[2]


def test_local_frame_editor_is_se3_equivariant():
    pair = backbone_pair()
    rotation = so3_exp(np.array([0.2, -0.1, 0.15]))
    translation = np.array([4.0, -2.0, 1.5])

    def transform(coords):
        return coords @ rotation.T + translation

    transformed_pair = StructurePair(
        "frames-transformed",
        pair.parent_sequence,
        pair.mutant_sequence,
        transform(pair.parent_coords),
        transform(pair.mutant_coords),
        pair.mutation_indices,
        pair.atom_names,
    )
    editor = LocalFrameDifferenceEditor(SequenceEndpoint())
    expected = transform(editor.predict(pair))
    actual = editor.predict(transformed_pair)
    assert np.allclose(actual, expected, atol=1e-6)
