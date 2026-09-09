import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ospedit.cli import main as eval_main
from ospedit.data import PairRecord, StructurePair, structure_pair_payload, write_manifest
from ospedit.train_cli import main


def test_train_cli_writes_checkpoint(tmp_path, monkeypatch):
    coords = np.array(
        [
            [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.0, 4.8, 0.0]],
        ]
    )
    pair = StructurePair("cli", "AA", "AY", coords, coords.copy(), (1,))
    dev_pair = StructurePair("cli-dev", "AA", "AY", coords, coords.copy(), (1,))
    teacher_pair = StructurePair("cli-teacher", "AA", "AY", coords, coords.copy(), (1,))
    manifest = tmp_path / "train.jsonl"
    write_manifest([
        PairRecord(pair, "parent", "family", "train"),
        PairRecord(dev_pair, "parent-dev", "family-dev", "dev"),
        PairRecord(teacher_pair, "parent-teacher", "family-teacher", "dev", label_source="teacher"),
    ], manifest)
    checkpoint = tmp_path / "student.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        ["ospedit-train", "--manifest", str(manifest), "--output", str(checkpoint), "--epochs", "1", "--hidden-dim", "32", "--blocks", "1", "--heads", "4", "--no-shuffle", "--eval-split", "dev", "--eval-batch-size", "2", "--gradient-clip-norm", "0.5", "--translation-scale", "2.0", "--rotation-scale", "0.25", "--max-normalized-delta", "0.5"],
    )
    main()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 1
    assert payload["config"]["record_count"] == 1
    assert payload["config"]["evaluation"]["split"] == "dev"
    assert payload["config"]["evaluation"]["runtime_summary"]["records"] == 1.0
    assert payload["config"]["evaluation"]["runtime_summary"]["batches"] == 1.0
    assert payload["config"]["evaluation"]["copy_parent_baseline"]["method"] == "copy_parent_baseline"
    assert payload["config"]["evaluation"]["copy_parent_baseline"]["runtime_summary"]["network_calls"] == 0.0
    assert isinstance(payload["config"]["evaluation"]["manifest_fingerprint"], str)
    assert payload["config"]["gradient_clip_norm"] == 0.5
    assert payload["config"]["translation_scale"] == 2.0
    assert payload["config"]["rotation_scale"] == 0.25
    assert payload["config"]["epochs"] == 1
    assert payload["config"]["last_run_epochs"] == 1
    assert payload["config"]["max_normalized_delta"] == 0.5
    assert payload["config"]["endpoint_group_balanced_loss"] is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ospedit-train",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "bad-endpoint-balance.pt"),
            "--resume",
            str(checkpoint),
            "--epochs",
            "1",
            "--translation-scale",
            "2.0",
            "--rotation-scale",
            "0.25",
            "--endpoint-group-balanced-loss",
        ],
    )
    with pytest.raises(SystemExit, match="endpoint_group_balanced_loss"):
        main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["ospedit-train", "--manifest", str(manifest), "--output", str(tmp_path / "bad-radius.pt"), "--resume", str(checkpoint), "--epochs", "1", "--translation-scale", "2.0", "--rotation-scale", "0.25", "--neighborhood-radius", "5.0"],
    )
    with pytest.raises(SystemExit, match="neighborhood_radius"):
        main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["ospedit-train", "--manifest", str(manifest), "--output", str(tmp_path / "bad-loss.pt"), "--resume", str(checkpoint), "--epochs", "1", "--translation-scale", "2.0", "--rotation-scale", "0.25", "--mutation-loss-weight", "1.0"],
    )
    with pytest.raises(SystemExit, match="mutation_loss_weight"):
        main()

    provenance_checkpoint = tmp_path / "provenance.pt"
    payload["config"]["teacher_cache_fingerprint"] = "different-cache"
    torch.save(payload, provenance_checkpoint)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ospedit-train", "--manifest", str(manifest), "--output", str(tmp_path / "bad-teacher.pt"), "--resume", str(provenance_checkpoint), "--epochs", "1", "--translation-scale", "2.0", "--rotation-scale", "0.25"],
    )
    with pytest.raises(SystemExit, match="teacher cache fingerprint"):
        main()

    changed_manifest = tmp_path / "changed.jsonl"
    write_manifest([
        PairRecord(pair, "parent", "family", "train"),
        PairRecord(dev_pair, "parent-dev", "family-dev", "dev"),
        PairRecord(teacher_pair, "parent-teacher", "family-teacher", "dev", label_source="teacher"),
        PairRecord(StructurePair("extra", "AA", "AY", coords, coords.copy(), (1,)), "extra-parent", "extra-family", "train"),
    ], changed_manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ospedit-train", "--manifest", str(changed_manifest), "--output", str(tmp_path / "bad-resume.pt"), "--resume", str(checkpoint), "--epochs", "1"],
    )
    with pytest.raises(SystemExit, match="fingerprint"):
        main()

    resumed = tmp_path / "student-resumed.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        ["ospedit-train", "--manifest", str(manifest), "--output", str(resumed), "--resume", str(checkpoint), "--epochs", "1", "--no-shuffle", "--translation-scale", "2.0", "--rotation-scale", "0.25"],
    )
    main()
    resumed_payload = torch.load(resumed, map_location="cpu", weights_only=False)
    assert resumed_payload["epoch"] == 2
    assert len(resumed_payload["history"]) == 2
    assert resumed_payload["config"]["epochs"] == 2
    assert resumed_payload["config"]["last_run_epochs"] == 1
    assert resumed_payload["config"]["max_normalized_delta"] == 0.5

    monkeypatch.setattr(
        sys,
        "argv",
        ["ospedit-train", "--manifest", str(manifest), "--output", str(tmp_path / "bad-scale.pt"), "--resume", str(checkpoint), "--epochs", "1", "--translation-scale", "2.0", "--rotation-scale", "0.25", "--geometry-features"],
    )
    with pytest.raises(ValueError, match="parent_dim"):
        main()

    pair_json = tmp_path / "pair.json"
    pair_json.write_text(json.dumps(structure_pair_payload(pair)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["ospedit-eval", str(pair_json), "--editor", "student", "--student-checkpoint", str(checkpoint)],
    )
    eval_main()
    from ospedit.cli import _student_editor
    loaded_editor = _student_editor(str(checkpoint), pair, "cpu")
    assert loaded_editor.translation_scale == 2.0
    assert loaded_editor.rotation_scale == 0.25

    ablated_checkpoint = tmp_path / "student-ablated.pt"
    payload["config"]["ablate_target_residue"] = True
    torch.save(payload, ablated_checkpoint)
    ablated_editor = _student_editor(str(ablated_checkpoint), pair, "cpu")
    assert ablated_editor.include_target_residue is False


def test_old_checkpoint_defaults_to_pre_position_behavior(tmp_path):
    from ospedit.cli import _student_editor
    from ospedit.student import ParentEditStudent
    from ospedit.student_training import save_student_checkpoint

    coords = np.zeros((2, 4, 3), dtype=float)
    pair = StructurePair("old", "AA", "AY", coords, coords.copy(), (1,))
    model = ParentEditStudent(parent_dim=16, hidden_dim=32, blocks=1, heads=4, use_positional_encoding=False)
    path = tmp_path / "old.pt"
    save_student_checkpoint(model, path, config={"parent_dim": 16, "hidden_dim": 32, "blocks": 1, "heads": 4})
    loaded = _student_editor(str(path), pair, "cpu")
    assert loaded.model.use_positional_encoding is False


def test_eval_cli_rejects_manifest_append_without_output(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ospedit-eval", "examples/toy_pair.json", "--manifest-append"])
    with pytest.raises(SystemExit):
        eval_main()
