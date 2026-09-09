import json

import numpy as np
import pytest

from ospedit import TeacherCache


def _cache(tmp_path):
    path = tmp_path / "index.json"
    np.savez_compressed(
        tmp_path / "pair.npz",
        **{
            "level_0.25_source_rotations": np.tile(np.eye(3), (2, 1, 1)),
            "level_0.25_source_origins": np.zeros((2, 3)),
            "level_0.25_target_rotations": np.tile(np.eye(3), (2, 1, 1)),
            "level_0.25_target_origins": np.zeros((2, 3)),
            "level_0.25_local_delta": np.ones((2, 6)),
            "level_0.25_valid": np.ones(2, dtype=np.uint8),
        },
    )
    path.write_text(json.dumps({"format": "ospedit.teacher_cache.v1", "split": "dev", "noise_levels": [0.25], "entries": [{"pair_id": "p", "file": "pair.npz", "length": 2}]}))
    return path


def test_teacher_cache_reads_and_validates_arrays(tmp_path):
    cache = TeacherCache.load(_cache(tmp_path), split="dev")
    arrays = cache.load_pair("p", 0.25)
    assert arrays["local_delta"].shape == (2, 6)
    assert arrays["valid"].dtype == bool
    assert np.allclose(cache.local_deltas(0.25)["p"][0], 1.0)


def test_teacher_cache_rejects_unknown_noise_level(tmp_path):
    cache = TeacherCache.load(_cache(tmp_path))
    with pytest.raises(KeyError, match="noise level"):
        cache.load_pair("p", 0.5)


def test_teacher_cache_rejects_tampered_file_when_checksum_is_recorded(tmp_path):
    index = _cache(tmp_path)
    from ospedit.data import file_sha256

    payload = json.loads(index.read_text())
    payload["entries"][0]["file_sha256"] = file_sha256(tmp_path / "pair.npz")
    index.write_text(json.dumps(payload))
    with (tmp_path / "pair.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        TeacherCache.load(index)
