import json

import numpy as np

from ospedit.data import PairRecord, StructurePair, write_manifest
from scripts.run_teacher_admission import main


def test_teacher_admission_cli_serializes_nan_remote_fields(tmp_path, monkeypatch):
    coords = np.asarray(
        [[[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]]
    )
    pair = StructurePair("all-mut", "A", "Y", coords, coords.copy(), (0,))
    manifest = tmp_path / "pairs.jsonl"
    write_manifest([PairRecord(pair, "p", "f", "dev")], manifest)
    factory = tmp_path / "factory.py"
    factory.write_text("class Endpoint:\n    def endpoint(self, coords, sequence, noise_level, *, noise_state):\n        import numpy as np\n        return np.tile(np.eye(3), (len(coords), 1, 1)), coords[:, 1].copy()\ndef build_endpoint():\n    return Endpoint()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    output = tmp_path / "admission.json"
    monkeypatch.setattr(
        "sys.argv",
        ["admission", "--manifest", str(manifest), "--endpoint-factory", "factory:build_endpoint", "--output", str(output)],
    )
    main()
    payload = json.loads(output.read_text())
    assert payload["records"][0]["diagnostic"]["levels"][0]["remote_translation_response"] is None
