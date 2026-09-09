import json

import numpy as np
import pytest

from ospedit.data import PairRecord, StructurePair, write_manifest
from scripts.run_teacher_admission import main as admission_main
from scripts.build_teacher_cache import build_cache


def _write_pair_manifest(tmp_path):
    coords = np.asarray(
        [
            [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.5, 3.8, 0.0]],
            [[-1.0, 8.1, 0.0], [0.0, 7.6, 0.0], [1.0, 7.6, 0.0], [1.5, 7.6, 0.0]],
        ]
    )
    pair = StructurePair("cache-pair", "AAA", "AYA", coords, coords.copy(), (1,))
    manifest = tmp_path / "pairs.jsonl"
    write_manifest([PairRecord(pair, "parent", "family", "dev")], manifest)
    return manifest


def test_teacher_cache_requires_admission_and_writes_local_delta(tmp_path, monkeypatch):
    manifest = _write_pair_manifest(tmp_path)
    factory = tmp_path / "cache_factory.py"
    factory.write_text(
        "class Endpoint:\n"
        "    def endpoint(self, coords, sequence, noise_level, *, noise_state):\n"
        "        import numpy as np\n"
        "        rotations = np.tile(np.eye(3), (len(coords), 1, 1))\n"
        "        origins = coords[:, 1].copy()\n"
        "        if sequence[1] == 'Y':\n"
        "            origins[1, 0] += 0.25\n"
        "        return rotations, origins\n"
        "def build_endpoint():\n"
        "    return Endpoint()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    admission = tmp_path / "admission.json"
    monkeypatch.setattr(
        "sys.argv",
         ["admission", "--manifest", str(manifest), "--endpoint-factory", "cache_factory:build_endpoint",
         "--output", str(admission), "--split", "dev", "--noise-levels", "0.25"],
    )
    admission_main()
    assert json.loads(admission.read_text())["ready_for_distillation"] is True

    output_dir = tmp_path / "cache"
    index = build_cache(manifest, admission, "cache_factory:build_endpoint", output_dir, split="dev", noise_levels=(0.25,))
    assert index["format"] == "ospedit.teacher_cache.v1"
    assert len(index["entries"]) == 1
    cache = np.load(output_dir / index["entries"][0]["file"])
    delta = cache["level_0.25_local_delta"]
    assert delta.shape == (3, 6)
    assert np.isclose(delta[1, 0], 0.25)
    assert np.allclose(delta[[0, 2]], 0.0)


def test_teacher_cache_can_require_direction_gate(tmp_path, monkeypatch):
    manifest = _write_pair_manifest(tmp_path)
    factory = tmp_path / "direction_factory.py"
    factory.write_text(
        "class Endpoint:\n"
        "    def endpoint(self, coords, sequence, noise_level, *, noise_state):\n"
        "        import numpy as np\n"
        "        origins = coords[:, 1].copy()\n"
        "        if sequence[1] == 'Y': origins[1, 0] += 0.25\n"
        "        return np.tile(np.eye(3), (len(coords), 1, 1)), origins\n"
        "def build_endpoint(): return Endpoint()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    admission = tmp_path / "admission.json"
    monkeypatch.setattr("sys.argv", ["admission", "--manifest", str(manifest), "--endpoint-factory", "direction_factory:build_endpoint", "--output", str(admission), "--split", "dev", "--noise-levels", "0.25"])
    admission_main()
    rejected = tmp_path / "rejected.json"
    rejected.write_text(json.dumps({"format": "ospedit.teacher_evaluation.v1", "direction_gate": {"accepted": False}}))
    with pytest.raises(ValueError, match="direction_gate"):
        build_cache(manifest, admission, "direction_factory:build_endpoint", tmp_path / "bad", split="dev", noise_levels=(0.25,), direction_report=rejected)

    malformed = tmp_path / "malformed-direction.json"
    malformed.write_text(json.dumps({"direction_gate": {"accepted": True}}))
    with pytest.raises(ValueError, match="teacher_evaluation.v1"):
        build_cache(manifest, admission, "direction_factory:build_endpoint", tmp_path / "malformed", split="dev", noise_levels=(0.25,), direction_report=malformed)
