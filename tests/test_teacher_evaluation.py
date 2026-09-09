import json

import numpy as np

from ospedit.data import PairRecord, StructurePair
from ospedit.teacher_cache import TeacherCache
from ospedit.teacher_evaluation import METRIC_NAMES, _macro_summary, evaluate_teacher_cache, evaluate_teacher_combination
from ospedit.student_data import target_local_delta


def test_teacher_evaluation_reports_perfect_delta_alignment(tmp_path):
    coords = np.asarray(
        [
            [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.5, 3.8, 0.0]],
        ]
    )
    mutant = coords.copy()
    mutant[1, :, 0] += 0.25
    pair = StructurePair("eval-pair", "AA", "AY", coords, mutant, (1,))
    record = PairRecord(pair, "parent", "family", "dev")
    delta, valid = target_local_delta(pair)
    identity = np.tile(np.eye(3), (2, 1, 1))
    np.savez_compressed(
        tmp_path / "pair.npz",
        **{
            "level_0.25_source_rotations": identity,
            "level_0.25_source_origins": coords[:, 1],
            "level_0.25_target_rotations": identity,
            "level_0.25_target_origins": mutant[:, 1],
            "level_0.25_local_delta": delta,
            "level_0.25_valid": valid.astype(np.uint8),
            "level_0.5_source_rotations": identity,
            "level_0.5_source_origins": coords[:, 1],
            "level_0.5_target_rotations": identity,
            "level_0.5_target_origins": mutant[:, 1],
            "level_0.5_local_delta": delta,
            "level_0.5_valid": valid.astype(np.uint8),
        },
    )
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"format": "ospedit.teacher_cache.v1", "noise_levels": [0.25, 0.5], "entries": [{"pair_id": "eval-pair", "file": "pair.npz", "length": 2}]}))
    cache = TeacherCache.load(index)
    report = evaluate_teacher_cache(cache, [record], 0.25)
    assert report["summary"]["mean_all_rmse"] == 0.0
    assert report["summary"]["mean_mutation_cosine"] == 1.0
    assert report["summary"]["mean_mutation_rmse"] == 0.0
    assert report["summary"]["mean_mutation_translation_rmse"] == 0.0
    assert report["summary"]["mean_mutation_rotation_rmse"] == 0.0
    assert report["parent_macro_summary"]["groups"] == 1
    assert report["family_macro_summary"]["groups"] == 1
    assert report["records"][0]["parent_id"] == "parent"
    assert "mean_transition_rmse" in report["summary"]
    combined = evaluate_teacher_combination(cache, [record], {0.25: 0.5, 0.5: 0.5})
    assert combined["summary"]["mean_mutation_cosine"] == 1.0
    assert combined["summary"]["mean_mutation_rmse"] == 0.0


def test_channel_metrics_keep_translation_and_rotation_separate(tmp_path):
    coords = np.asarray(
        [
            [[-1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            [[-1.0, 4.3, 0.0], [0.0, 3.8, 0.0], [1.0, 3.8, 0.0], [1.5, 3.8, 0.0]],
        ]
    )
    mutant = coords.copy()
    mutant[1, :, 0] += 1.0
    pair = StructurePair("channel-pair", "AA", "AY", coords, mutant, (1,))
    record = PairRecord(pair, "parent", "family", "dev")
    delta, valid = target_local_delta(pair)
    identity = np.tile(np.eye(3), (2, 1, 1))
    teacher_delta = delta.copy()
    teacher_delta[1, 0] += 0.5
    np.savez_compressed(
        tmp_path / "channel.npz",
        **{
            "level_0.25_source_rotations": identity,
            "level_0.25_source_origins": coords[:, 1],
            "level_0.25_target_rotations": identity,
            "level_0.25_target_origins": mutant[:, 1],
            "level_0.25_local_delta": teacher_delta,
            "level_0.25_valid": valid.astype(np.uint8),
        },
    )
    index = tmp_path / "channel-index.json"
    index.write_text(json.dumps({"format": "ospedit.teacher_cache.v1", "noise_levels": [0.25], "entries": [{"pair_id": "channel-pair", "file": "channel.npz", "length": 2}]}))
    report = evaluate_teacher_cache(TeacherCache.load(index), [record], 0.25)
    assert np.isclose(report["summary"]["mean_mutation_translation_rmse"], 0.5 / np.sqrt(3.0))
    assert report["summary"]["mean_mutation_rotation_rmse"] == 0.0


def test_parent_macro_does_not_overweight_repeated_parent_records():
    def row(parent_id, cosine):
        values = {name: 0.0 for name in METRIC_NAMES}
        values.update({"parent_id": parent_id, "mutation_cosine": cosine})
        return values

    summary = _macro_summary([row("p1", 1.0), row("p1", 1.0), row("p2", -1.0)], "parent_id")

    assert summary["groups"] == 2
    assert summary["mean_mutation_cosine"] == 0.0
