import json

import numpy as np

from ospedit.data import PairRecord, StructurePair
from ospedit.experiment import evaluate_editor_suite, flatten_suite_reports, suite_payload, write_suite_csv, write_suite_report
from ospedit.models import CopyParentEditor


def test_suite_payload_and_writer_are_strict_json(tmp_path):
    coords = np.zeros((2, 1, 3))
    record = PairRecord(StructurePair("p", "AA", "AY", coords, coords, (1,)), "parent", "family", "dev")
    suite = evaluate_editor_suite(
        [record], {"copy": CopyParentEditor()}, split="dev", run_metadata={"seed": 3}
    )
    payload = suite_payload(suite)
    encoded = json.dumps(payload, allow_nan=False)
    assert '"copy"' in encoded
    assert isinstance(payload["manifest_fingerprint"], str)
    assert payload["run_metadata"] == {"seed": 3}
    path = tmp_path / "nested" / "suite.json"
    write_suite_report(suite, path)
    assert json.loads(path.read_text()) == payload

    rows = flatten_suite_reports(suite)
    assert rows[0]["method"] == "copy"
    assert rows[0]["records"] == 1
    assert isinstance(rows[0]["manifest_fingerprint"], str)
    assert rows[0]["config_seed"] == 3
    csv_path = tmp_path / "nested" / "suite.csv"
    write_suite_csv(suite, csv_path)
    csv_text = csv_path.read_text()
    assert "method" in csv_text and "copy" in csv_text
