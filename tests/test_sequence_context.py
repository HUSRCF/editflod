import json

import numpy as np
import pytest

from ospedit.sequence_context import (
    SEQUENCE_CONTEXT_FORMAT,
    SequenceContextCache,
    write_sequence_context_cache,
)


def test_sequence_context_cache_round_trip(tmp_path):
    path = tmp_path / "context.npz"
    metadata = write_sequence_context_cache(
        path,
        {
            "AA": np.arange(6, dtype=np.float32).reshape(2, 3),
            "AY": np.ones((2, 3), dtype=np.float32),
        },
        model_id="test/model",
        model_revision="revision",
    )

    cache = SequenceContextCache.load(path)

    assert metadata["format"] == SEQUENCE_CONTEXT_FORMAT
    assert cache.embedding_dim == 3
    assert cache.model_id == "test/model"
    assert cache.fingerprint == metadata["fingerprint"]
    assert np.array_equal(cache.get("AA"), np.arange(6).reshape(2, 3))
    assert cache.get("AA").flags.writeable is False
    with pytest.raises(KeyError, match="missing sequence"):
        cache.get("AC")


def test_sequence_context_cache_rejects_tampered_metadata(tmp_path):
    path = tmp_path / "context.npz"
    write_sequence_context_cache(
        path,
        {"AA": np.zeros((2, 2), dtype=np.float32)},
        model_id="test/model",
    )
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["model_id"] = "other/model"
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="fingerprint"):
        SequenceContextCache.load(path)


def test_sequence_context_writer_rejects_length_mismatch(tmp_path):
    with pytest.raises(ValueError, match="invalid embedding shape"):
        write_sequence_context_cache(
            tmp_path / "bad.npz",
            {"AAA": np.zeros((2, 4), dtype=np.float32)},
            model_id="test/model",
        )
