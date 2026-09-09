import json

import numpy as np

from ospedit.data import PairRecord, StructurePair
from ospedit.experiment import evaluate_parent_workloads, flatten_parent_workload_reports, parent_workload_payload, parent_workloads, write_parent_workload_csv, write_parent_workload_report
from ospedit.models import CopyParentEditor


def test_parent_workload_report_round_trips_as_strict_json(tmp_path):
    coords = np.zeros((2, 1, 3))
    records = [
        PairRecord(StructurePair(f"a{i}", "AA", "AY", coords, coords, (1,)), "parent", "family", "dev")
        for i in range(2)
    ]
    reports = evaluate_parent_workloads(parent_workloads(records, (1, 2)), CopyParentEditor(), method="copy")
    payload = parent_workload_payload(reports)
    json.dumps(payload, allow_nan=False)
    path = tmp_path / "nested" / "workloads.json"
    write_parent_workload_report(reports, path)
    assert json.loads(path.read_text()) == payload
    assert set(payload["parents"]["parent"]) == {"1", "2"}
    assert isinstance(payload["parents"]["parent"]["1"]["manifest_fingerprint"], str)
    rows = flatten_parent_workload_reports(reports)
    assert [row["candidate_count"] for row in rows] == [1, 2]
    assert rows[0]["records"] == 1.0
    assert rows[0]["split"] == ""
    assert isinstance(rows[0]["manifest_fingerprint"], str)
    csv_path = tmp_path / "nested" / "workloads.csv"
    write_parent_workload_csv(reports, csv_path)
    lines = csv_path.read_text().splitlines()
    assert "candidate_count" in lines[0]
    assert len(lines) == 3
