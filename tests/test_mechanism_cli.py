import json
import sys

import numpy as np
import pytest

from ospedit.data import PairRecord, StructurePair, write_manifest
from scripts.run_mechanism_grid import main


def test_mechanism_grid_cli_writes_optional_csv(tmp_path, monkeypatch):
    coords = np.array(
        [
            [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.0, 4.8, 0.0]],
        ]
    )
    pair = StructurePair("cli-grid", "AA", "AY", coords, coords.copy(), (1,))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([PairRecord(pair, "parent", "family", "dev")], manifest)
    factory = tmp_path / "mechanism_factory.py"
    factory.write_text(
        "import numpy as np\n"
        "class Endpoint:\n"
        "    def endpoint(self, coords, sequence, noise_level, noise_state=None):\n"
        "        return np.tile(np.eye(3), (len(coords), 1, 1)), coords[:, 1].copy()\n"
        "def build_endpoint():\n"
        "    return Endpoint()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    output = tmp_path / "nested" / "grid.json"
    csv_output = tmp_path / "nested" / "grid.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mechanism_grid",
            "--manifest",
            str(manifest),
            "--endpoint-factory",
            "mechanism_factory:build_endpoint",
            "--output",
            str(output),
            "--csv-output",
            str(csv_output),
            "--split",
            "dev",
            "--methods",
            "C0_copy_parent,C3_two_noise_shared_difference",
        ],
    )
    main()
    payload = json.loads(output.read_text())
    assert set(payload["methods"]) == {"C0_copy_parent", "C3_two_noise_shared_difference"}
    lines = csv_output.read_text().splitlines()
    assert len(lines) == 3
    assert "method" in lines[0]
    assert any("C0_copy_parent" in line for line in lines[1:])


def test_mechanism_grid_cli_applies_frozen_config(tmp_path, monkeypatch):
    coords = np.array([
        [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.0, 4.8, 0.0]],
    ])
    pair = StructurePair("frozen", "AA", "AY", coords, coords.copy(), (1,))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([PairRecord(pair, "parent", "family", "test")], manifest)
    factory = tmp_path / "frozen_factory.py"
    factory.write_text(
        "import numpy as np\n"
        "class Endpoint:\n"
        "    def endpoint(self, coords, sequence, noise_level, noise_state=None):\n"
        "        return np.tile(np.eye(3), (len(coords), 1, 1)), coords[:, 1].copy()\n"
        "def build_endpoint():\n"
        "    return Endpoint()\n"
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "format": "ospedit.mechanism_config.v1",
        "source_manifest": str(manifest),
        "method": "C3_two_noise_shared_difference",
        "run_id": 7,
        "source_manifest_fingerprint": "dev-fingerprint",
        "config": {"noise_level": 0.2, "noise_levels": [0.1, 0.6], "noise_weights": [0.4, 0.5], "difference_step_size": 0.03, "translation_scale": 0.4, "rotation_scale": 0.5},
    }))
    monkeypatch.syspath_prepend(str(tmp_path))
    output = tmp_path / "frozen-grid.json"
    monkeypatch.setattr(sys, "argv", [
        "run_mechanism_grid", "--manifest", str(manifest), "--endpoint-factory", "frozen_factory:build_endpoint",
        "--output", str(output), "--split", "test", "--frozen-config", str(config),
    ])
    main()
    payload = json.loads(output.read_text())
    metadata = payload["run_metadata"]
    assert metadata["noise_level"] == 0.2
    assert metadata["difference_step_size"] == 0.03
    assert metadata["noise_levels"] == [0.1, 0.6]
    assert metadata["noise_weights"] == [0.4, 0.5]
    assert metadata["frozen_config_run_id"] == 7
    assert metadata["frozen_config_method"] == "C3_two_noise_shared_difference"


def test_mechanism_grid_rejects_frozen_config_for_other_manifest(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "format": "ospedit.mechanism_config.v1",
        "source_manifest": str(tmp_path / "dev.jsonl"),
        "config": {"noise_level": 0.2, "difference_step_size": 0.03, "translation_scale": 1.0, "rotation_scale": 1.0},
    }))
    monkeypatch.setattr(sys, "argv", [
        "run_mechanism_grid", "--manifest", str(tmp_path / "test.jsonl"),
        "--endpoint-factory", "missing:factory", "--output", str(tmp_path / "out.json"),
        "--frozen-config", str(config),
    ])
    with pytest.raises(SystemExit, match="source_manifest does not match"):
        main()


def test_mechanism_grid_requires_frozen_method_in_filter(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "format": "ospedit.mechanism_config.v1",
        "method": "C3_two_noise_shared_difference",
        "config": {"noise_level": 0.2, "difference_step_size": 0.03, "translation_scale": 1.0, "rotation_scale": 1.0},
    }))
    monkeypatch.setattr(sys, "argv", [
        "run_mechanism_grid", "--manifest", "missing.jsonl", "--endpoint-factory", "missing:factory",
        "--output", str(tmp_path / "out.json"), "--methods", "C0_copy_parent", "--frozen-config", str(config),
    ])
    with pytest.raises(SystemExit, match="must include the method"):
        main()
