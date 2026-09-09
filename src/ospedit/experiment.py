from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable
import json
import csv
from pathlib import Path

import numpy as np

from .data import PairRecord, StructurePair, manifest_fingerprint
from .metrics import evaluate_pair
from .models import ConditionalDifferenceEditor, CopyParentEditor, Editor, EndpointModel, FieldModel, IndependentNoiseLocalFrameDifferenceEditor, LocalFrameDifferenceEditor, MultiNoiseLocalFrameDifferenceEditor, MutationNeighborhoodDifferenceEditor, RepeatedSingleNoiseLocalFrameDifferenceEditor, StudentEditor, TargetUpdateEditor
from .runtime import RuntimeStats
from .data import json_safe


@dataclass(frozen=True)
class EvaluationResult:
    method: str
    metrics: dict[str, float | str]
    runtime: RuntimeStats


@dataclass(frozen=True)
class ManifestEvaluation:
    method: str
    split: str | None
    records: list[dict[str, object]]
    family_summary: dict[str, dict[str, float]]
    runtime_summary: dict[str, float]
    manifest_fingerprint: str | None = None


@dataclass(frozen=True)
class SuiteEvaluation:
    split: str | None
    methods: dict[str, ManifestEvaluation]
    manifest_fingerprint: str | None = None
    run_metadata: dict[str, object] = field(default_factory=dict)


def group_records_by_parent(
    records: Iterable[PairRecord], *, split: str | None = None
) -> dict[str, list[PairRecord]]:
    """Group audited records by parent for multi-candidate cost experiments."""
    groups: dict[str, list[PairRecord]] = {}
    for record in records:
        if split is not None and record.split != split:
            continue
        groups.setdefault(record.parent_id, []).append(record)
    return groups


def parent_workloads(
    records: Iterable[PairRecord],
    candidate_counts: Iterable[int] = (1, 32, 128),
    *,
    split: str | None = None,
    seed: int = 0,
    require_full: bool = True,
) -> dict[str, dict[int, list[PairRecord]]]:
    """Build deterministic same-parent candidate workloads for cost studies."""
    counts = tuple(int(count) for count in candidate_counts)
    if any(count <= 0 for count in counts):
        raise ValueError("candidate_counts must contain only positive values")
    if len(set(counts)) != len(counts):
        raise ValueError("candidate_counts must not contain duplicates")
    groups = group_records_by_parent(records, split=split)
    output: dict[str, dict[int, list[PairRecord]]] = {}
    for parent_id, candidates in groups.items():
        if require_full and any(len(candidates) < count for count in counts):
            continue
        order = np.random.default_rng(seed).permutation(len(candidates))
        output[parent_id] = {
            count: [candidates[int(index)] for index in order[:count]]
            for count in counts
            if len(candidates) >= count
        }
    return output


def evaluate_parent_workloads(
    workloads: dict[str, dict[int, list[PairRecord]]],
    editor: Editor,
    *,
    batch_size: int = 1,
    method: str | None = None,
) -> dict[str, dict[int, ManifestEvaluation]]:
    """Evaluate each deterministic parent/candidate workload independently."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return {
        parent_id: {
            count: evaluate_manifest_batched(
                records,
                editor,
                batch_size=batch_size,
                method=method,
                split=None,
            )
            for count, records in sorted(by_count.items())
        }
        for parent_id, by_count in workloads.items()
    }


def parent_workload_payload(
    reports: dict[str, dict[int, ManifestEvaluation]],
) -> dict[str, object]:
    """Convert parent workload reports into strict-JSON-safe dictionaries."""
    return json_safe({
        "parents": {
            parent_id: {
                str(count): {
                    "method": report.method,
                    "split": report.split,
                    "records": report.records,
                    "family_summary": report.family_summary,
                    "runtime_summary": report.runtime_summary,
                    "manifest_fingerprint": report.manifest_fingerprint,
                }
                for count, report in sorted(by_count.items())
            }
            for parent_id, by_count in reports.items()
        }
    })


def write_parent_workload_report(
    reports: dict[str, dict[int, ManifestEvaluation]], path: str | Path
) -> None:
    """Write deterministic JSON for 1/32/128 parent workload experiments."""
    payload = json.dumps(parent_workload_payload(reports), indent=2, sort_keys=True, allow_nan=False)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload + "\n")


def flatten_parent_workload_reports(
    reports: dict[str, dict[int, ManifestEvaluation]],
) -> list[dict[str, object]]:
    """Flatten workload reports into one row per parent and candidate count."""
    rows: list[dict[str, object]] = []
    for parent_id, by_count in reports.items():
        for count, report in sorted(by_count.items()):
            row: dict[str, object] = {
                "parent_id": parent_id,
                "candidate_count": int(count),
                "method": report.method,
                "split": report.split or "",
                "manifest_fingerprint": report.manifest_fingerprint or "",
                **report.runtime_summary,
            }
            if report.family_summary:
                summaries = list(report.family_summary.values())
                keys = summaries[0].keys()
                def mean_or_nan(key: str) -> float:
                    values = np.asarray([summaries[index][key] for index in range(len(summaries))], dtype=float)
                    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

                row.update({f"mean_{key}": mean_or_nan(key) for key in keys})
            rows.append(row)
    return rows


def write_parent_workload_csv(
    reports: dict[str, dict[int, ManifestEvaluation]], path: str | Path
) -> None:
    """Write flattened workload reports as a stable CSV table."""
    rows = flatten_parent_workload_reports(reports)
    fieldnames = sorted({key for row in rows for key in row})
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = {
                key: ("" if isinstance(value, float) and not np.isfinite(value) else value)
                for key, value in row.items()
            }
            writer.writerow(clean)


def suite_payload(suite: SuiteEvaluation) -> dict[str, object]:
    """Convert a suite report into strict-JSON-safe nested dictionaries."""
    return json_safe({
        "split": suite.split,
        "manifest_fingerprint": suite.manifest_fingerprint,
        "run_metadata": suite.run_metadata,
        "methods": {
            name: {
                "method": report.method,
                "split": report.split,
                "records": report.records,
                "family_summary": report.family_summary,
                "runtime_summary": report.runtime_summary,
                "manifest_fingerprint": report.manifest_fingerprint,
            }
            for name, report in suite.methods.items()
        },
    })


def write_suite_report(suite: SuiteEvaluation, path: str | Path) -> None:
    """Write a deterministic, strict JSON mechanism report."""
    payload = json.dumps(suite_payload(suite), indent=2, sort_keys=True, allow_nan=False)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload + "\n")


def flatten_suite_reports(suite: SuiteEvaluation) -> list[dict[str, object]]:
    """Flatten one suite into one CSV-friendly summary row per method."""
    rows: list[dict[str, object]] = []
    for name, report in suite.methods.items():
        row: dict[str, object] = {
            "method": name,
            "split": report.split or "",
            "records": len(report.records),
            "manifest_fingerprint": report.manifest_fingerprint or "",
            **report.runtime_summary,
        }
        for key, value in suite.run_metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"config_{key}"] = value
            else:
                row[f"config_{key}"] = json.dumps(json_safe(value), sort_keys=True, allow_nan=False)
        if report.family_summary:
            summaries = list(report.family_summary.values())
            for key in summaries[0].keys():
                values = np.asarray([summary[key] for summary in summaries], dtype=float)
                row[f"mean_{key}"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        rows.append(row)
    return rows


def write_suite_csv(suite: SuiteEvaluation, path: str | Path) -> None:
    """Write a stable one-row-per-method CSV summary for a mechanism suite."""
    rows = flatten_suite_reports(suite)
    fieldnames = sorted({key for row in rows for key in row})
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: ("" if isinstance(value, float) and not np.isfinite(value) else value)
                for key, value in row.items()
            })


def build_mechanism_editors(
    *,
    field_model: FieldModel | None = None,
    endpoint_model: EndpointModel | None = None,
    noise_level: float = 0.5,
    noise_levels: tuple[float, ...] = (0.25, 0.75),
    multi_noise_weights: tuple[float, ...] | None = None,
    target_step_size: float = 1.0,
    difference_step_size: float = 1.0,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
    neighborhood_radius: float | None = None,
) -> dict[str, Editor]:
    """Build a consistently named C0-C3 mechanism comparison suite.

    External baselines are intentionally not inserted here; callers can add
    them to the returned mapping before calling ``evaluate_editor_suite``.
    """
    editors: dict[str, Editor] = {"C0_copy_parent": CopyParentEditor()}
    endpoint_translation_scale = translation_scale * difference_step_size
    endpoint_rotation_scale = rotation_scale * difference_step_size
    if field_model is not None:
        editors["C1_target_only"] = TargetUpdateEditor(field_model, target_step_size, noise_level)
        editors["C2_single_noise_difference"] = ConditionalDifferenceEditor(
            field_model, difference_step_size, noise_level
        )
    if endpoint_model is not None:
        editors["C2_single_noise_local_frame_difference"] = LocalFrameDifferenceEditor(
            endpoint_model, endpoint_translation_scale, endpoint_rotation_scale, noise_level
        )
        editors["C3_two_noise_shared_difference"] = MultiNoiseLocalFrameDifferenceEditor(
            endpoint_model,
            tuple(noise_levels),
            tuple(multi_noise_weights) if multi_noise_weights is not None else None,
            endpoint_translation_scale,
            endpoint_rotation_scale,
        )
        editors["C4_repeated_single_noise_matched_budget"] = RepeatedSingleNoiseLocalFrameDifferenceEditor(
            endpoint_model, noise_level, 2, endpoint_translation_scale, endpoint_rotation_scale
        )
        editors["C5_independent_noise_difference"] = IndependentNoiseLocalFrameDifferenceEditor(
            endpoint_model, noise_level, endpoint_translation_scale, endpoint_rotation_scale
        )
        if neighborhood_radius is not None:
            editors["C6_mutation_neighborhood_difference"] = MutationNeighborhoodDifferenceEditor(
                endpoint_model,
                endpoint_translation_scale,
                endpoint_rotation_scale,
                noise_level,
                neighborhood_radius,
            )
    return editors


def evaluate_manifest_batched(
    records: Iterable[PairRecord],
    editor: Editor,
    *,
    batch_size: int = 1,
    method: str | None = None,
    split: str | None = None,
) -> ManifestEvaluation:
    """Evaluate a batch-capable editor while preserving per-pair metrics."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not hasattr(editor, "predict_batch"):
        return evaluate_manifest(records, editor, method=method, split=split)
    selected = [record for record in records if split is None or record.split == split]
    if not selected:
        raise ValueError(f"manifest contains no records for split={split!r}")
    rows: list[dict[str, object]] = []
    cache_before = _cache_counters(editor)
    total_seconds = 0.0
    batch_count = 0
    conditional_batch_count = 0
    import time
    for start in range(0, len(selected), batch_size):
        group = selected[start:start + batch_size]
        began = time.perf_counter()
        predictions = editor.predict_batch([record.pair for record in group])
        elapsed = time.perf_counter() - began
        total_seconds += elapsed
        batch_count += 1
        if any(record.pair.parent_sequence != record.pair.mutant_sequence for record in group):
            conditional_batch_count += 1
        for record, prediction in zip(group, predictions, strict=True):
            rows.append({
                "pair_id": record.pair.pair_id,
                "parent_id": record.parent_id,
                "family_id": record.family_id,
                "split": record.split,
                "method": method or type(editor).__name__,
                "metrics": evaluate_pair(record.pair, prediction),
                "runtime": {"batch_size": len(group), "batch_seconds": elapsed},
            })
    # Quality aggregation uses the same row order as the regular evaluator.
    metric_results = [type("Result", (), {"metrics": row["metrics"]})() for row in rows]
    family_summary = aggregate_by_family(selected, metric_results)
    return ManifestEvaluation(
        method=method or type(editor).__name__,
        split=split,
        records=rows,
        family_summary=family_summary,
        runtime_summary={
            "records": float(len(selected)),
            "batches": float(batch_count),
            "conditional_batches": float(conditional_batch_count),
            "batch_size": float(batch_size),
            "total_seconds": total_seconds,
            "mean_seconds": total_seconds / len(selected),
            "network_calls": float(conditional_batch_count * _condition_branch_count(editor)),
            "sequence_encoder_calls": float(conditional_batch_count * _sequence_encoder_branch_count(editor)),
            "condition_branches": float(conditional_batch_count * _condition_branch_count(editor)),
            "structure_updates": float(sum(record.pair.parent_sequence != record.pair.mutant_sequence for record in selected)),
            **_cache_runtime_fields(editor, cache_before),
        },
        manifest_fingerprint=manifest_fingerprint(selected),
    )


def _cache_counters(editor: Editor) -> tuple[int, int, int] | None:
    cache = getattr(editor, "parent_cache", None)
    if cache is None:
        return None
    return int(getattr(cache, "hits", 0)), int(getattr(cache, "misses", 0)), len(cache)


def _cache_runtime_fields(editor: Editor, before: tuple[int, int, int] | None = None) -> dict[str, float]:
    counters = _cache_counters(editor)
    if counters is None:
        return {}
    previous_hits, previous_misses, _ = before or (0, 0, 0)
    hits, misses, entries = counters
    return {
        "parent_cache_hits": float(hits - previous_hits),
        "parent_cache_misses": float(misses - previous_misses),
        "parent_cache_entries": float(entries),
    }


def _condition_branch_count(editor: Editor) -> int:
    """Return conditional model evaluations per edited sample."""
    configured = getattr(editor, "condition_branches_per_edit", None)
    if configured is not None:
        return int(configured)
    if isinstance(editor, CopyParentEditor):
        return 0
    if isinstance(editor, StudentEditor):
        return 1
    if isinstance(editor, MultiNoiseLocalFrameDifferenceEditor):
        return 2 * len(editor.noise_levels)
    if isinstance(editor, RepeatedSingleNoiseLocalFrameDifferenceEditor):
        return 2 * editor.repeats
    if isinstance(editor, IndependentNoiseLocalFrameDifferenceEditor):
        return 2
    if isinstance(editor, (ConditionalDifferenceEditor, LocalFrameDifferenceEditor, MutationNeighborhoodDifferenceEditor)):
        return 2
    if isinstance(editor, TargetUpdateEditor):
        return 1
    return 0


def _sequence_encoder_branch_count(editor: Editor) -> int:
    """Return sequence-model encoder evaluations per edited sample."""
    configured = getattr(editor, "sequence_encoder_calls_per_edit", None)
    if configured is not None:
        return int(configured)
    if isinstance(editor, StudentEditor):
        # The one-pass student consumes discrete edit features; it does not
        # invoke the large pretrained sequence encoder at inference time.
        return 0
    return _condition_branch_count(editor)


def evaluate_editor(pair: StructurePair, editor: Editor, method: str | None = None) -> EvaluationResult:
    no_edit = pair.parent_sequence == pair.mutant_sequence
    stats = RuntimeStats(structure_updates=0 if isinstance(editor, CopyParentEditor) or no_edit else 1)
    if no_edit:
        pass
    elif isinstance(editor, StudentEditor):
        stats.network_calls = 1
        stats.sequence_encoder_calls = 0
        stats.condition_branches = 1
    elif isinstance(editor, MultiNoiseLocalFrameDifferenceEditor):
        branches = 2 * len(editor.noise_levels)
        stats.network_calls = branches
        stats.sequence_encoder_calls = branches
        stats.condition_branches = branches
    elif isinstance(editor, IndependentNoiseLocalFrameDifferenceEditor):
        stats.network_calls = 2
        stats.sequence_encoder_calls = 2
        stats.condition_branches = 2
    elif isinstance(editor, RepeatedSingleNoiseLocalFrameDifferenceEditor):
        branches = 2 * editor.repeats
        stats.network_calls = branches
        stats.sequence_encoder_calls = branches
        stats.condition_branches = branches
    elif isinstance(editor, (ConditionalDifferenceEditor, LocalFrameDifferenceEditor)):
        stats.network_calls = 2
        stats.sequence_encoder_calls = 2
        stats.condition_branches = 2
    elif isinstance(editor, TargetUpdateEditor):
        stats.network_calls = 1
        stats.sequence_encoder_calls = 1
        stats.condition_branches = 1
    else:
        stats.network_calls = int(getattr(editor, "network_calls_per_edit", 0))
        stats.sequence_encoder_calls = _sequence_encoder_branch_count(editor)
        stats.condition_branches = _condition_branch_count(editor)

    cache_before = _cache_counters(editor)
    start = perf_counter()
    prediction = editor.predict(pair)
    stats.total_seconds = perf_counter() - start
    stats.extras.update(_cache_runtime_fields(editor, cache_before))
    runtime = getattr(editor, "last_runtime", None)
    if isinstance(runtime, dict):
        stats.extras.update(runtime)
        if getattr(editor, "use_reported_runtime", False) and "end_to_end_seconds" in runtime:
            stats.total_seconds = float(runtime["end_to_end_seconds"])
    name = method or type(editor).__name__
    return EvaluationResult(name, evaluate_pair(pair, prediction), stats)


def evaluate_records(records: Iterable[PairRecord], editor: Editor, method: str | None = None) -> list[EvaluationResult]:
    return [evaluate_editor(record.pair, editor, method) for record in records]


def aggregate_by_family(records: Iterable[PairRecord], results: Iterable[EvaluationResult]) -> dict[str, dict[str, float]]:
    """Macro-average pair metrics within families, then report family rows."""
    groups: dict[str, list[dict[str, float | str]]] = {}
    for record, result in zip(records, results, strict=True):
        groups.setdefault(record.family_id, []).append(result.metrics)
    output: dict[str, dict[str, float]] = {}
    for family, rows in groups.items():
        numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
        output[family] = {}
        for key in numeric_keys:
            values = np.asarray([float(row[key]) for row in rows])
            output[family][key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return output


def evaluate_manifest(
    records: Iterable[PairRecord],
    editor: Editor,
    method: str | None = None,
    split: str | None = None,
) -> ManifestEvaluation:
    selected = [record for record in records if split is None or record.split == split]
    if not selected:
        raise ValueError(f"manifest contains no records for split={split!r}")
    results = evaluate_records(selected, editor, method)
    rows: list[dict[str, object]] = []
    total_seconds = 0.0
    for record, result in zip(selected, results, strict=True):
        rows.append(
            {
                "pair_id": record.pair.pair_id,
                "parent_id": record.parent_id,
                "family_id": record.family_id,
                "split": record.split,
                "method": result.method,
                "metrics": result.metrics,
                "runtime": result.runtime.as_dict(),
            }
        )
        total_seconds += result.runtime.total_seconds
    cache_totals: dict[str, float] = {}
    for result in results:
        for key, value in result.runtime.extras.items():
            if isinstance(value, (int, float)):
                numeric = float(value)
                if key.startswith("peak_") or key.endswith("_rmsd"):
                    cache_totals[key] = max(cache_totals.get(key, float("-inf")), numeric)
                else:
                    cache_totals[key] = cache_totals.get(key, 0.0) + numeric
    return ManifestEvaluation(
        method=method or type(editor).__name__,
        split=split,
        records=rows,
        family_summary=aggregate_by_family(selected, results),
        runtime_summary={
            "records": float(len(selected)),
            "total_seconds": total_seconds,
            "mean_seconds": total_seconds / len(selected),
            "network_calls": float(sum(int(result.runtime.network_calls) for result in results)),
            "sequence_encoder_calls": float(sum(int(result.runtime.sequence_encoder_calls) for result in results)),
            "condition_branches": float(sum(int(result.runtime.condition_branches) for result in results)),
            "structure_updates": float(sum(int(result.runtime.structure_updates) for result in results)),
            **cache_totals,
        },
        manifest_fingerprint=manifest_fingerprint(selected),
    )


def evaluate_editor_suite(
    records: Iterable[PairRecord],
    editors: dict[str, Editor],
    split: str | None = None,
    batch_size: int = 1,
    run_metadata: dict[str, object] | None = None,
) -> SuiteEvaluation:
    """Evaluate all methods on the same records, optionally in batches."""
    if not editors:
        raise ValueError("editors must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    materialized = list(records)
    methods = {
        name: evaluate_manifest_batched(materialized, editor, batch_size=batch_size, method=name, split=split)
        for name, editor in editors.items()
    }
    selected = [record for record in materialized if split is None or record.split == split]
    return SuiteEvaluation(
        split=split,
        methods=methods,
        manifest_fingerprint=manifest_fingerprint(selected),
        run_metadata=dict(run_metadata or {}),
    )
