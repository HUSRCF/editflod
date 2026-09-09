from scripts.run_student_architecture_sweep import (
    _comparison,
    _family_macro,
    aggregate_runs,
)


def test_architecture_sweep_family_macro_ignores_missing_cosines():
    families = {
        "a": {"local_backbone_error": 1.0, "distance_change_cosine": None},
        "b": {"local_backbone_error": 3.0, "distance_change_cosine": 0.5},
    }

    summary = _family_macro(families)

    assert summary["local_backbone_error"] == 2.0
    assert summary["distance_change_cosine"] == 0.5


def test_architecture_sweep_comparison_and_seed_aggregate():
    student = {"a": {"local_backbone_error": 1.0}}
    copied = {"a": {"local_backbone_error": 2.0}}
    comparison = _comparison(student, copied)
    runs = [
        {
            "architecture": "transformer",
            "seed": seed,
            "parameter_count": 10,
            "train": comparison,
            "dev": comparison,
        }
        for seed in (0, 1)
    ]

    aggregate = aggregate_runs(runs)

    local = aggregate["transformer"]["dev"]["student_minus_copy"][
        "local_backbone_error"
    ]
    assert local == {"count": 2, "mean": -1.0, "std": 0.0}
    assert aggregate["transformer"]["parameter_count"] == [10]


def test_architecture_sweep_comparison_preserves_undefined_copy_cosine():
    student = {"a": {"distance_change_cosine": 0.25}}
    copied = {"a": {"distance_change_cosine": None}}

    comparison = _comparison(student, copied)

    assert comparison["student_family_macro"]["distance_change_cosine"] == 0.25
    assert comparison["student_minus_copy"]["distance_change_cosine"] is None
