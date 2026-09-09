from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from ospedit.data import PairRecord, StructurePair, write_manifest
from scripts.run_mechanism_sweep import _select_records, main


def test_mechanism_sweep_writes_all_configurations(tmp_path, monkeypatch):
    residue = np.array(
        [[0.0, 1.2, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 1.0, 0.0]],
        dtype=float,
    )
    coords = np.repeat(residue[None, :, :], 2, axis=0)
    manifest = tmp_path / "pairs.jsonl"
    write_manifest([PairRecord(StructurePair("p", "AA", "AY", coords, coords.copy(), (1,)), "parent", "family", "dev")], manifest)
    factory = tmp_path / "factory.py"
    factory.write_text(
        "class Endpoint:\n"
        "    def endpoint(self, coords, sequence, noise_level, noise_state=None):\n"
        "        import numpy as np\n"
        "        return np.tile(np.eye(3), (len(coords), 1, 1)), coords[:, 1].copy()\n"
        "def build_endpoint():\n"
        "    return Endpoint()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    output = tmp_path / "nested" / "sweep.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_mechanism_sweep",
            "--manifest", str(manifest),
            "--endpoint-factory", "factory:build_endpoint",
            "--output", str(output),
            "--noise-levels", "0.2,0.4",
            "--difference-step-sizes", "0.1,0.2",
            "--methods", "C0_copy_parent,C3_two_noise_shared_difference",
            "--max-records", "1",
        ],
    )
    main()
    payload = json.loads(output.read_text())
    assert payload["format"] == "ospedit.mechanism_sweep.v1"
    assert payload["status"] == "completed"
    assert len(payload["runs"]) == 4
    assert all(len(run["rows"]) == 2 for run in payload["runs"])
    assert {run["run_id"] for run in payload["runs"]} == {0, 1, 2, 3}
    assert payload["runs"][0]["config"]["noise_levels"] == [0.25, 0.75]
    assert payload["runs"][0]["rows"][1]["method"] == "C3_two_noise_shared_difference"
    assert payload["runs"][0]["rows"][0]["records"] == 1

    partial = dict(payload)
    partial["status"] = "running"
    partial["runs"] = payload["runs"][:1]
    partial["expected_configurations"] = 99
    output.write_text(json.dumps(partial))
    monkeypatch.setattr("sys.argv", [*sys.argv, "--resume"])
    with pytest.raises(SystemExit, match="grid size"):
        main()
    partial["expected_configurations"] = 4
    output.write_text(json.dumps(partial))
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_mechanism_sweep",
            "--manifest", str(manifest),
            "--endpoint-factory", "factory:build_endpoint",
            "--output", str(output),
            "--noise-levels", "0.2,0.4",
            "--difference-step-sizes", "0.1,0.2",
            "--methods", "C0_copy_parent",
            "--max-records", "1",
            "--resume",
        ],
    )
    with pytest.raises(SystemExit, match="method filter"):
        main()
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_mechanism_sweep",
            "--manifest", str(manifest),
            "--endpoint-factory", "factory:build_endpoint",
            "--output", str(output),
            "--noise-levels", "0.2,0.4",
            "--difference-step-sizes", "0.1,0.2",
            "--methods", "C0_copy_parent,C3_two_noise_shared_difference",
            "--max-records", "1",
            "--resume",
        ],
    )
    main()
    resumed = json.loads(output.read_text())
    assert resumed["status"] == "completed"
    assert len(resumed["runs"]) == 4


def test_family_round_robin_selection_is_deterministic():
    records = [
        type("Record", (), {"family_id": "a", "pair_id": "a1"})(),
        type("Record", (), {"family_id": "a", "pair_id": "a2"})(),
        type("Record", (), {"family_id": "b", "pair_id": "b1"})(),
    ]
    selected = _select_records(records, 3, "family_round_robin")
    assert [record.pair_id for record in selected] == ["a1", "b1", "a2"]
