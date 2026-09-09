import json
from pathlib import Path
import sys

import numpy as np
import pytest

from examples.esmfold_predict import _parse_backbone
from examples.premut_predict import _backbone_indices
from ospedit.data import PairRecord, StructurePair, file_sha256
from ospedit.experiment import evaluate_editor
from ospedit.external_baselines import ESMFoldEditor, PreMutEditor


def _record(tmp_path: Path) -> PairRecord:
    source = tmp_path / "parent.pdb"
    source.write_text("test parent\n")
    parent = np.arange(3 * 4 * 3, dtype=float).reshape(3, 4, 3) / 10.0
    mutant = parent.copy()
    mutant[1, :, 2] += 0.2
    pair = StructurePair("pair/one", "AAA", "AYA", parent, mutant, (1,))
    return PairRecord(
        pair,
        parent_id="parent",
        family_id="family",
        split="test",
        source_file=str(source),
        source_chain="A",
        source_checksum=file_sha256(source),
    )


def _editor(
    tmp_path: Path, record: PairRecord, *, replay_generation_cost: bool = False
) -> PreMutEditor:
    root = tmp_path / "upstream"
    root.mkdir()
    checkpoint = root / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    runner = tmp_path / "runner.py"
    runner.write_text("raise SystemExit('cache should be reused')\n")
    return PreMutEditor(
        [record],
        upstream_root=root,
        checkpoint=checkpoint,
        python_executable=sys.executable,
        prediction_dir=tmp_path / "predictions",
        runner=runner,
        replay_generation_cost=replay_generation_cost,
    )


def _write_cache(editor: PreMutEditor, record: PairRecord, input_backbone: np.ndarray) -> None:
    metadata = {
        **editor._expected_metadata(record),
        "runtime": {"inference_seconds": 0.25},
    }
    prediction = record.pair.parent_coords.copy()
    prediction[1, :, 0] += 0.1
    np.savez_compressed(
        editor._cache_path(record),
        prediction=prediction,
        input_backbone=input_backbone,
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_premut_editor_validates_cache_and_reports_external_cost(tmp_path):
    record = _record(tmp_path)
    editor = _editor(tmp_path, record)
    _write_cache(editor, record, record.pair.parent_coords)

    result = evaluate_editor(record.pair, editor, method="premut")

    assert result.runtime.network_calls == 1
    assert result.runtime.sequence_encoder_calls == 0
    assert result.runtime.condition_branches == 1
    assert result.runtime.extras["cache_hit"] == 1.0
    assert result.runtime.extras["inference_seconds"] == 0.25


def test_premut_editor_rejects_mismatched_reconstructed_input(tmp_path):
    record = _record(tmp_path)
    editor = _editor(tmp_path, record)
    _write_cache(editor, record, record.pair.parent_coords + 1.0)

    with pytest.raises(ValueError, match="reconstructed input differs"):
        editor.predict(record.pair)


def test_premut_editor_rejects_changed_source_file_before_cache_reuse(tmp_path):
    record = _record(tmp_path)
    editor = _editor(tmp_path, record)
    _write_cache(editor, record, record.pair.parent_coords)
    Path(record.source_file).write_text("changed parent\n")

    with pytest.raises(ValueError, match="source checksum mismatch"):
        editor.predict(record.pair)


def test_premut_editor_can_replay_persistent_generation_cost(tmp_path):
    record = _record(tmp_path)
    editor = _editor(tmp_path, record, replay_generation_cost=True)
    _write_cache(editor, record, record.pair.parent_coords)

    result = evaluate_editor(record.pair, editor)

    assert result.runtime.total_seconds == pytest.approx(0.25)


def test_premut_backbone_indices_preserve_residue_atom_order():
    assert _backbone_indices(
        ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O", "CB", "CG"]
    ).tolist() == [[0, 1, 2, 3], [5, 6, 7, 8]]


def test_esmfold_editor_checks_identity_and_aligns_to_parent(tmp_path):
    record = _record(tmp_path)
    runner = tmp_path / "esmfold_runner.py"
    runner.write_text("runner\n")
    editor = ESMFoldEditor(
        prediction_dir=tmp_path / "esmfold",
        mode="zero_extra_recycles",
        model_identity={"model.pt": "abc"},
        runner_checksum=file_sha256(runner),
    )
    Path(editor.prediction_dir).mkdir()
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    prediction = record.pair.parent_coords @ rotation.T + np.asarray([4.0, -3.0, 2.0])
    metadata = {
        **editor.expected_metadata(record.pair),
        "runtime": {
            "model_load_seconds_amortized": 1.0,
            "inference_seconds": 2.0,
            "pdb_conversion_seconds": 0.5,
        },
    }
    np.savez_compressed(
        editor.cache_path(record.pair),
        prediction=prediction,
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    result = evaluate_editor(record.pair, editor, method="esmfold")

    assert np.allclose(editor.predict(record.pair), record.pair.parent_coords)
    assert result.runtime.total_seconds >= 3.5
    assert result.runtime.network_calls == 1
    assert result.runtime.sequence_encoder_calls == 1


def test_esmfold_pdb_parser_preserves_backbone_order_and_sequence():
    lines = []
    serial = 1
    for residue_number, residue_name in ((1, "ALA"), (2, "TYR")):
        for atom, x in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0), ("CB", 4.0)):
            lines.append(
                f"ATOM  {serial:5d} {atom:>4s} {residue_name:>3s} A{residue_number:4d}"
                f"    {x + residue_number:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
            )
            serial += 1

    sequence, coordinates = _parse_backbone("\n".join(lines) + "\n")

    assert sequence == "AY"
    assert coordinates.shape == (2, 4, 3)
    assert coordinates[:, :, 0].tolist() == [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]
