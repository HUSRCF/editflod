import numpy as np

from ospedit.experiment import build_mechanism_editors
from ospedit.models import ConditionalDifferenceEditor, MultiNoiseLocalFrameDifferenceEditor, TargetUpdateEditor


class Field:
    def field(self, coords, sequence, noise_level):
        return np.zeros_like(coords)


class Endpoint:
    def endpoint(self, coords, sequence, noise_level):
        return np.tile(np.eye(3), (len(coords), 1, 1)), coords[:, 1].copy()


def test_build_mechanism_editors_has_stable_c0_c3_names_and_configuration():
    editors = build_mechanism_editors(field_model=Field(), endpoint_model=Endpoint(), noise_level=0.4)
    assert list(editors) == [
        "C0_copy_parent",
        "C1_target_only",
        "C2_single_noise_difference",
        "C2_single_noise_local_frame_difference",
        "C3_two_noise_shared_difference",
        "C4_repeated_single_noise_matched_budget",
        "C5_independent_noise_difference",
    ]
    assert isinstance(editors["C1_target_only"], TargetUpdateEditor)
    assert isinstance(editors["C2_single_noise_difference"], ConditionalDifferenceEditor)
    assert isinstance(editors["C3_two_noise_shared_difference"], MultiNoiseLocalFrameDifferenceEditor)
    assert editors["C1_target_only"].noise_level == 0.4
