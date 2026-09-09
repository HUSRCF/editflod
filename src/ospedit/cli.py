from __future__ import annotations

import argparse
import json

from .data import (
    StructurePair,
    append_manifest,
    json_safe,
    load_manifest,
    pair_record_from_structures,
    pair_parsed_structures,
    parse_structure,
    structure_pair_payload,
    validate_manifest,
    verify_record_checksums,
    write_manifest,
)
from .experiment import evaluate_editor, evaluate_manifest_batched
from .metrics import METRIC_SCHEMA_VERSION
from .models import CopyParentEditor, StudentEditor
from .student import HybridSpatialGraphStudent, ParentEditStudent, SpatialGraphStudent
from .student_data import parent_local_features
from .student_training import (
    load_student_checkpoint,
    validate_student_checkpoint_config,
)


def _student_editor(
    checkpoint: str,
    pair: StructurePair,
    device: str,
    update_scale: float = 1.0,
    output_localization_radius: float | None = None,
    output_localization_transition: float | None = None,
) -> StudentEditor:
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise ValueError("student evaluation requires torch; install ospedit[torch]") from error
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = dict(payload.get("config", {}))
    include_geometry = bool(config.get("geometry_features", False))
    include_biochemical = bool(config.get("biochemical_edit_features", False))
    include_target_residue = not bool(config.get("ablate_target_residue", False))
    checkpoint_localization_radius = config.get("target_localization_radius")
    localization_radius = output_localization_radius if output_localization_radius is not None else checkpoint_localization_radius
    localization_transition = (
        output_localization_transition
        if output_localization_transition is not None
        else float(config.get("target_localization_transition", 5.0))
    )
    edit_dim = int(config.get("edit_dim", 48 if include_biochemical else 41))
    # Checkpoints written before positional encoding was introduced must keep
    # their original behavior instead of silently changing at evaluation.
    use_positional_encoding = not bool(config["no_positional_encoding"]) if "no_positional_encoding" in config else False
    validate_student_checkpoint_config(
        config,
        parent_dim=parent_local_features(pair, include_geometry=include_geometry).shape[-1],
        edit_dim=edit_dim,
    )
    architecture = str(config.get("student_architecture", "transformer"))
    model: object
    if architecture == "spatial_graph_global":
        model = HybridSpatialGraphStudent(
            parent_dim=int(
                config.get(
                    "parent_dim",
                    parent_local_features(pair, include_geometry=include_geometry).shape[-1],
                )
            ),
            edit_dim=edit_dim,
            hidden_dim=int(config.get("hidden_dim", 128)),
            graph_blocks=int(config.get("blocks", 2)),
            global_blocks=int(config.get("global_blocks", 1)),
            heads=int(config.get("heads", 8)),
            max_normalized_delta=config.get("max_normalized_delta"),
        )
    elif architecture == "spatial_graph":
        model = SpatialGraphStudent(
            parent_dim=int(
                config.get(
                    "parent_dim",
                    parent_local_features(pair, include_geometry=include_geometry).shape[-1],
                )
            ),
            edit_dim=edit_dim,
            hidden_dim=int(config.get("hidden_dim", 128)),
            blocks=int(config.get("blocks", 4)),
            max_normalized_delta=config.get("max_normalized_delta"),
        )
    else:
        model = ParentEditStudent(
            parent_dim=int(
                config.get(
                    "parent_dim",
                    parent_local_features(pair, include_geometry=include_geometry).shape[-1],
                )
            ),
            edit_dim=edit_dim,
            hidden_dim=int(config.get("hidden_dim", 256)),
            blocks=int(config.get("blocks", 4)),
            heads=int(config.get("heads", 8)),
            max_normalized_delta=config.get("max_normalized_delta"),
            use_positional_encoding=use_positional_encoding,
        )
    load_student_checkpoint(model, checkpoint, map_location=device)
    return StudentEditor(
        model,
        device=device,
        translation_scale=float(config.get("translation_scale", 1.0)),
        rotation_scale=float(config.get("rotation_scale", 1.0)),
        include_geometry=include_geometry,
        include_spatial_graph=architecture in {"spatial_graph", "spatial_graph_global"},
        spatial_neighbors=int(config.get("spatial_neighbors", 24)),
        update_scale=update_scale,
        include_biochemical=include_biochemical,
        include_target_residue=include_target_residue,
        output_localization_radius=localization_radius,
        output_localization_transition=localization_transition,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a reference-conditioned structure editor")
    parser.add_argument("pair", nargs="?", help="JSON StructurePair file")
    parser.add_argument("--audit-manifest", help="Validate a JSON/JSONL pair manifest")
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Verify source/target checksums during manifest audit",
    )
    parser.add_argument(
        "--allow-split-overlap",
        action="store_true",
        help="Allow family/parent overlap only for an explicitly diagnostic split",
    )
    parser.add_argument("--manifest", help="Evaluate all records in a JSON/JSONL manifest")
    parser.add_argument("--results-output", help="Write batch evaluation JSON to this path")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Student manifest inference batch size",
    )
    parser.add_argument("--student-checkpoint", help="Student .pt checkpoint for --editor student")
    parser.add_argument("--student-update-scale", type=float, default=1.0)
    parser.add_argument("--student-output-localization-radius", type=float)
    parser.add_argument("--student-output-localization-transition", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parent-structure", help="Parent PDB/mmCIF path; use with --mutant-structure")
    parser.add_argument("--mutant-structure", help="Mutant PDB/mmCIF path; use with --parent-structure")
    parser.add_argument("--parent-chain", default="A")
    parser.add_argument("--mutant-chain", default="A")
    parser.add_argument("--pair-id", default="structure_pair")
    parser.add_argument("--parent-id", default=None)
    parser.add_argument("--family-id", default=None)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--eval-split", choices=("train", "dev", "test"), default=None)
    parser.add_argument(
        "--label-source",
        choices=("experimental", "synthetic", "teacher"),
        default="experimental",
    )
    parser.add_argument("--output", help="Output JSON path for a generated structure pair")
    parser.add_argument(
        "--manifest-output",
        help="Write generated pair as one JSON/JSONL manifest record",
    )
    parser.add_argument(
        "--manifest-append",
        action="store_true",
        help="Append generated record to --manifest-output",
    )
    parser.add_argument(
        "--editor",
        choices=("copy_source_backbone", "student"),
        default="copy_source_backbone",
    )
    args = parser.parse_args()
    editor: StudentEditor | CopyParentEditor
    if args.manifest_append and not args.manifest_output:
        parser.error("--manifest-append requires --manifest-output")
    if args.audit_manifest:
        records = load_manifest(args.audit_manifest)
        errors = validate_manifest(records, allow_split_overlap=args.allow_split_overlap)
        if args.verify_checksums:
            errors.extend(error for record in records for error in verify_record_checksums(record))
        print(json.dumps({"records": len(records), "errors": errors}, indent=2, allow_nan=False))
        raise SystemExit(1 if errors else 0)
    if args.manifest:
        records = load_manifest(args.manifest)
        errors = validate_manifest(records, allow_split_overlap=args.allow_split_overlap)
        if args.verify_checksums:
            errors.extend(error for record in records for error in verify_record_checksums(record))
        if errors:
            parser.error("manifest audit failed: " + "; ".join(errors))
        if args.editor == "student":
            if not args.student_checkpoint:
                parser.error("--editor student requires --student-checkpoint")
            candidates = [record for record in records if args.eval_split is None or record.split == args.eval_split]
            if not candidates:
                parser.error(f"manifest contains no records for eval split={args.eval_split!r}")
            editor = _student_editor(
                args.student_checkpoint,
                candidates[0].pair,
                args.device,
                args.student_update_scale,
                args.student_output_localization_radius,
                args.student_output_localization_transition,
            )
        else:
            editor = CopyParentEditor()
        report = evaluate_manifest_batched(
            records,
            editor,
            batch_size=args.batch_size,
            method=args.editor,
            split=args.eval_split,
        )
        payload = json.dumps(
            json_safe(
                {
                    "metric_schema": METRIC_SCHEMA_VERSION,
                    "method": report.method,
                    "split": report.split,
                    "records": report.records,
                    "family_summary": report.family_summary,
                    "runtime_summary": report.runtime_summary,
                    "manifest_fingerprint": report.manifest_fingerprint,
                }
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        if args.results_output:
            from pathlib import Path

            Path(args.results_output).write_text(payload + "\n")
        else:
            print(payload)
        return
    if bool(args.parent_structure) != bool(args.mutant_structure):
        parser.error("--parent-structure and --mutant-structure must be supplied together")
    if args.parent_structure:
        parent = parse_structure(args.parent_structure, args.parent_chain)
        mutant = parse_structure(args.mutant_structure, args.mutant_chain)
        if args.manifest_output:
            record = pair_record_from_structures(
                parent,
                mutant,
                pair_id=args.pair_id,
                parent_id=args.parent_id or args.pair_id,
                family_id=args.family_id or args.parent_id or args.pair_id,
                split=args.split,
                label_source=args.label_source,
            )
            if args.manifest_append:
                append_manifest([record], args.manifest_output)
            else:
                write_manifest([record], args.manifest_output)
            return
        pair = pair_parsed_structures(
            parent,
            mutant,
            args.pair_id,
        )
        payload = json.dumps(structure_pair_payload(pair), indent=2, allow_nan=False)
        if args.output:
            from pathlib import Path

            Path(args.output).write_text(payload + "\n")
        else:
            print(payload)
        return
    if not args.pair:
        parser.error("pair is required unless --audit-manifest is used")
    pair = StructurePair.from_json(args.pair)
    if args.editor == "student":
        if not args.student_checkpoint:
            parser.error("--editor student requires --student-checkpoint")
        editor = _student_editor(
            args.student_checkpoint,
            pair,
            args.device,
            args.student_update_scale,
            args.student_output_localization_radius,
            args.student_output_localization_transition,
        )
    else:
        editor = CopyParentEditor()
    result = evaluate_editor(pair, editor, args.editor)
    print(
        json.dumps(
            json_safe(
                {
                    "metric_schema": METRIC_SCHEMA_VERSION,
                    "method": result.method,
                    "metrics": result.metrics,
                    "runtime": result.runtime.as_dict(),
                }
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
