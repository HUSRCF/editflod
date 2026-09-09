import pickle

import numpy as np

from ospedit.data import PairRecord, StructurePair, write_manifest
from scripts.build_protocol_v02_manifest import build_protocol_manifest, write_compact_index


def _record(pair_id: str, parent: str, family: str, sequence: str) -> PairRecord:
    residue = np.array([[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    coords = np.repeat(residue[None], len(sequence), axis=0)
    mutant = sequence[:-1] + ("Y" if sequence[-1] != "Y" else "A")
    pair = StructurePair(pair_id, sequence, mutant, coords, coords.copy(), (len(sequence) - 1,))
    return PairRecord(pair, parent, family, "train", source_file=f"{parent}.pdb", target_file=f"{pair_id}.pdb")


def test_protocol_builder_filters_and_keeps_related_records_together(tmp_path):
    rows = [
        _record("a", "1AAA_A", "fa", "A" * 64),
        _record("b", "2AAA_A", "fb", "A" * 63 + "C"),
        _record("c", "3AAA_A", "fc", "C" * 64),
        _record("d", "4AAA_A", "fd", "D" * 64),
    ]
    manifest = tmp_path / "input.jsonl"
    write_manifest(rows, manifest)
    cluster = tmp_path / "cluster_dict"
    with cluster.open("wb") as handle:
        pickle.dump({"cluster": ["1AAA_A", "2AAA_A"]}, handle)

    output, report = build_protocol_manifest(
        [manifest],
        cluster_dicts=[cluster],
        min_sequence_identity=0.95,
        min_sequence_coverage=1.0,
    )

    indexed = {record.pair.pair_id: record for record in output}
    assert indexed["a"].family_id == indexed["b"].family_id
    assert indexed["a"].split == indexed["b"].split
    assert indexed["a"].split == "train"
    assert len({record.split for record in output}) == 3
    assert report["selected_records"] == 4


def test_protocol_builder_filters_length_and_nonexperimental_records(tmp_path):
    short = _record("short", "short", "short", "A" * 10)
    kept = _record("kept", "kept", "kept", "C" * 64)
    second = _record("second", "second", "second", "D" * 64)
    third = _record("third", "third", "third", "E" * 64)
    synthetic = replace_label(_record("synthetic", "synthetic", "synthetic", "D" * 64))
    manifest = tmp_path / "input.jsonl"
    write_manifest([short, kept, second, third, synthetic], manifest)

    output, report = build_protocol_manifest(
        [manifest], min_sequence_identity=1.0, min_sequence_coverage=1.0
    )

    assert {record.pair.pair_id for record in output} == {"kept", "second", "third"}
    assert report["filtered_records"] == 2


def replace_label(record: PairRecord) -> PairRecord:
    return PairRecord(
        record.pair,
        record.parent_id,
        record.family_id,
        record.split,
        label_source="synthetic",
        source_file=record.source_file,
        target_file=record.target_file,
    )


def test_protocol_compact_index_omits_coordinates(tmp_path):
    destination = tmp_path / "index.csv"
    write_compact_index([_record("a", "1AAA_A", "family", "A" * 64)], destination)
    text = destination.read_text()
    assert "pair_id,parent_id,family_id,split,length" in text
    assert "a,1AAA_A,family,train,64" in text
    assert "parent_coords" not in text
