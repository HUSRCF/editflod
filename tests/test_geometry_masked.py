import numpy as np

from ospedit.geometry import residue_frames_masked


def test_masked_frames_keep_valid_residues_when_one_is_degenerate():
    coords = np.asarray([
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ])
    rotations, origins, valid = residue_frames_masked(coords, ("N", "CA", "C", "O"))
    assert valid.tolist() == [True, False]
    assert np.allclose(origins[1], [0.0, 0.0, 0.0])
    assert np.allclose(rotations[1], np.eye(3))
