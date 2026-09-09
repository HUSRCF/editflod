import numpy as np
import torch

from ospedit.adapters import FoldFlow2EndpointAdapter
from ospedit.noise import make_shared_noise


class FakeFoldFlow:
    def __init__(self):
        self.conditioned = False

    def conditional_generation(self):
        self.conditioned = True

    @property
    def is_conditional_generation(self):
        return self.conditioned

    def eval(self):
        self.conditioned = False
        return self

    def __call__(self, batch):
        assert self.conditioned
        return {"rigids": torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]]])}


def test_foldflow_adapter_converts_rigids_and_preserves_noise_contract():
    seen = []

    def build(coords, sequence, noise_level, noise_state):
        seen.append(noise_state)
        return {"length": torch.tensor([len(sequence)])}

    adapter = FoldFlow2EndpointAdapter(FakeFoldFlow(), build)
    state = make_shared_noise((1, 1, 3), 0.2, 0)
    rotations, origins = adapter.endpoint(np.zeros((1, 1, 3)), "A", 0.2, noise_state=state)
    assert rotations.shape == (1, 3, 3)
    assert np.allclose(rotations[0], np.eye(3))
    assert np.allclose(origins, [[1.0, 2.0, 3.0]])
    assert seen == [state]


def test_foldflow_quaternion_order_is_scalar_first():
    def build(coords, sequence, noise_level, noise_state):
        return {}

    class Rotating(FakeFoldFlow):
        def __call__(self, batch):
            s = 2**-0.5
            return {"rigids": torch.tensor([[[s, 0.0, 0.0, s, 0.0, 0.0, 0.0]]])}

    adapter = FoldFlow2EndpointAdapter(Rotating(), build)
    rotations, _ = adapter.endpoint(
        np.zeros((1, 1, 3)), "A", 0.0, noise_state=make_shared_noise((1, 1, 3), 0.0, 0)
    )
    assert np.allclose(rotations[0], [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_foldflow_adapter_recovers_conditioning_after_external_eval():
    model = FakeFoldFlow()
    adapter = FoldFlow2EndpointAdapter(model, lambda coords, sequence, level, state: {})
    model.eval()
    state = make_shared_noise((1, 1, 3), 0.0, 0)
    adapter.endpoint(np.zeros((1, 1, 3)), "A", 0.0, noise_state=state)
    assert model.conditioned
