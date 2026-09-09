import numpy as np
import torch

from ospedit.foldflow_batch import FoldFlow2BatchBuilder, FoldFlow2MarginalConverter, backbone_to_rigids, make_foldflow_reference_rigid_sampler, make_forward_marginal_sampler, openfold_rigid_from_tensor7, sequence_to_aatype
from ospedit.noise import make_shared_noise


def test_sequence_to_aatype_uses_foldflow_order_and_unknown_index():
    encoded = sequence_to_aatype("ARNDX")
    assert encoded.tolist() == [0, 1, 2, 3, 20]


def test_foldflow_batch_builder_creates_minimal_batched_inputs():
    seen = []

    def coords_to_rigids(coords, noise_state):
        seen.append(noise_state)
        return np.tile(np.asarray([1, 0, 0, 0, 0, 0, 0], dtype=float), (len(coords), 1))

    builder = FoldFlow2BatchBuilder(coords_to_rigids)
    state = make_shared_noise((2, 4, 3), 0.3, 11)
    batch = builder(np.zeros((2, 4, 3)), "AY", 0.3, state)
    assert set(batch) == {"aatype", "chain_idx", "seq_idx", "res_mask", "fixed_mask", "sc_ca_t", "t", "rigids_t"}
    assert batch["rigids_t"].shape == (1, 2, 7)
    assert batch["aatype"].dtype == torch.long
    assert batch["res_mask"].dtype == torch.bool
    assert batch["seq_idx"].tolist() == [[1, 2]]
    assert seen == [state]


def test_foldflow_batch_builder_rejects_malformed_sequence_encoder_output():
    builder = FoldFlow2BatchBuilder(
        lambda coords, noise_state: np.tile([1, 0, 0, 0, 0, 0, 0], (len(coords), 1)),
        sequence_encoder=lambda sequence: np.zeros((1, len(sequence)), dtype=np.int64),
    )
    state = make_shared_noise((2, 4, 3), 0.0, 0)
    try:
        builder(np.zeros((2, 4, 3)), "AY", 0.0, state)
    except ValueError as error:
        assert "sequence_encoder" in str(error)
    else:
        raise AssertionError("expected sequence encoder shape validation")


def test_foldflow_batch_builder_rejects_noise_level_mismatch():
    builder = FoldFlow2BatchBuilder(
        lambda coords, noise_state: np.tile([1, 0, 0, 0, 0, 0, 0], (len(coords), 1))
    )
    state = make_shared_noise((2, 4, 3), 0.2, 0)
    try:
        builder(np.zeros((2, 4, 3)), "AY", 0.3, state)
    except ValueError as error:
        assert "noise_level" in str(error)
    else:
        raise AssertionError("expected noise level validation")


def test_foldflow_batch_builder_accepts_explicit_residue_mask():
    builder = FoldFlow2BatchBuilder(
        lambda coords, noise_state: np.tile([1, 0, 0, 0, 0, 0, 0], (len(coords), 1)),
        residue_mask_builder=lambda coords: np.array([True, False]),
    )
    state = make_shared_noise((2, 4, 3), 0.0, 0)
    batch = builder(np.zeros((2, 4, 3)), "AY", 0.0, state)
    assert batch["res_mask"].tolist() == [[True, False]]


def test_foldflow_batch_builder_infers_standard_frame_mask():
    builder = FoldFlow2BatchBuilder(
        lambda coords, noise_state: np.tile([1, 0, 0, 0, 0, 0, 0], (len(coords), 1))
    )
    coords = np.zeros((2, 4, 3), dtype=float)
    coords[0] = np.asarray([[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    state = make_shared_noise(coords.shape, 0.0, 0)
    batch = builder(coords, "AY", 0.0, state)
    assert batch["res_mask"].tolist() == [[True, False]]


def test_foldflow_batch_builder_rejects_bad_residue_mask_shape():
    builder = FoldFlow2BatchBuilder(
        lambda coords, noise_state: np.tile([1, 0, 0, 0, 0, 0, 0], (len(coords), 1)),
        residue_mask_builder=lambda coords: np.ones((1, len(coords)), dtype=bool),
    )
    state = make_shared_noise((2, 4, 3), 0.0, 0)
    try:
        builder(np.zeros((2, 4, 3)), "AY", 0.0, state)
    except ValueError as error:
        assert "residue_mask_builder" in str(error)
    else:
        raise AssertionError("expected residue mask shape validation")


def test_backbone_to_rigids_returns_openfold_style_identity_and_frame():
    coords = np.asarray([
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
    ])
    rigids = backbone_to_rigids(coords)
    assert rigids.shape == (1, 7)
    assert np.allclose(rigids[0, :4], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(rigids[0, 4:], [0.0, 0.0, 0.0])


def test_backbone_to_rigids_masks_degenerate_residue():
    coords = np.zeros((1, 4, 3))
    rigids = backbone_to_rigids(coords)
    assert np.array_equal(rigids[0, :4], [1.0, 0.0, 0.0, 0.0])


def test_marginal_converter_forwards_shared_noise_and_accepts_model_dict():
    coords = np.asarray([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]])
    seen = []

    def build_rigid(clean):
        return {"clean": clean}

    def sample(clean, level, state):
        seen.append((clean, level, state))
        return {"rigids_t": np.tile([1.0, 0.0, 0.0, 0.0, 2.0, 3.0, 4.0], (1, 1, 1))}

    converter = FoldFlow2MarginalConverter(build_rigid, sample)
    state = make_shared_noise(coords.shape, 0.4, 9)
    result = converter(coords, state)
    assert result.shape == (1, 7)
    assert np.allclose(result[0, 4:], [2.0, 3.0, 4.0])
    assert seen[0][1] == 0.4
    assert seen[0][2] is state


def test_forward_marginal_sampler_matches_foldflow_call_signature():
    calls = []

    class Matcher:
        def forward_marginal(self, clean, *, t, rigids_1, as_tensor_7):
            calls.append((clean, t, rigids_1, as_tensor_7))
            return {"rigids_t": np.zeros((1, 7))}

    clean = object()
    target = object()
    state = make_shared_noise((1, 1, 3), 0.6, 3)
    sampler = make_forward_marginal_sampler(Matcher(), lambda value, noise: target if noise is state else None)
    result = sampler(clean, 0.6, state)
    assert result["rigids_t"].shape == (1, 7)
    assert calls == [(clean, 0.6, target, True)]


def test_foldflow_reference_sampler_reuses_seed_and_restores_rng():
    calls = []

    class Matcher:
        def sample_ref(self, *, n_samples, as_tensor_7):
            calls.append((n_samples, as_tensor_7, float(np.random.normal()), float(torch.rand(1))))
            return calls[-1]

    np.random.seed(91)
    torch.manual_seed(91)
    expected_np = np.random.normal()
    expected_torch = float(torch.rand(1))
    np.random.seed(91)
    torch.manual_seed(91)
    sampler = make_foldflow_reference_rigid_sampler(Matcher())
    state = make_shared_noise((1, 2, 3), 0.4, 22)
    first = sampler(type("Rigid", (), {"shape": (1, 2)})(), state)
    second = sampler(type("Rigid", (), {"shape": (1, 2)})(), state)
    assert first == second
    assert calls[0][:2] == (2, False)
    assert np.isclose(np.random.normal(), expected_np)
    assert np.isclose(float(torch.rand(1)), expected_torch)


def test_foldflow_reference_sampler_unwraps_official_mapping():
    target = object()

    class Matcher:
        def sample_ref(self, *, n_samples, as_tensor_7):
            assert (n_samples, as_tensor_7) == (2, False)
            return {"rigids_t": target}

    sampler = make_foldflow_reference_rigid_sampler(Matcher())
    state = make_shared_noise((1, 2, 3), 0.4, 22)
    assert sampler(type("Rigid", (), {"shape": (1, 2)})(), state) is target


def test_foldflow_reference_sampler_adds_missing_batch_axis():
    class Sample:
        shape = (2,)

        def unsqueeze(self, dim):
            assert dim == 0
            return "batched"

    class Matcher:
        def sample_ref(self, *, n_samples, as_tensor_7):
            return Sample()

    sampler = make_foldflow_reference_rigid_sampler(Matcher())
    state = make_shared_noise((1, 2, 3), 0.4, 22)
    assert sampler(type("Rigid", (), {"shape": (1, 2)})(), state) == "batched"


def test_openfold_builder_reports_optional_dependency_boundary():
    try:
        openfold_rigid_from_tensor7(np.zeros((1, 7)))
    except RuntimeError as error:
        assert "OpenFold" in str(error)
    else:
        # This branch is valid in a fully provisioned FoldFlow environment.
        rigid = openfold_rigid_from_tensor7(np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
        assert tuple(rigid.get_trans().shape) == (1, 1, 3)
