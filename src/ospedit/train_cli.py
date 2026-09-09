from __future__ import annotations

import argparse
import json
import random
from typing import Any

import numpy as np

from .data import (
    json_safe,
    load_manifest,
    manifest_fingerprint,
    validate_manifest,
    verify_record_checksums,
)
from .experiment import evaluate_manifest_batched
from .models import CopyParentEditor, StudentEditor
from .student import HybridSpatialGraphStudent, ParentEditStudent, SpatialGraphStudent
from .student_data import parent_local_features
from .student_training import (
    LOSS_SCHEMA_VERSION,
    load_student_checkpoint,
    save_student_checkpoint,
    train_records,
    validate_student_checkpoint_config,
)
from .teacher_cache import TeacherCache


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the supervised one-pass parent-edit student")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="Output .pt checkpoint")
    parser.add_argument("--resume", help="Resume model/optimizer/history from a checkpoint")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--eval-split", choices=("train", "dev", "test"), default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument(
        "--student-architecture",
        choices=("transformer", "spatial_graph", "spatial_graph_global"),
        default="transformer",
    )
    parser.add_argument("--spatial-neighbors", type=int, default=24)
    parser.add_argument("--global-blocks", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="Angstroms represented by one predicted translation unit",
    )
    parser.add_argument(
        "--rotation-scale",
        type=float,
        default=1.0,
        help="Radians represented by one predicted rotation unit",
    )
    parser.add_argument(
        "--delta-norm-weight",
        type=float,
        default=0.0,
        help="Optional penalty on predicted normalized update magnitude",
    )
    parser.add_argument(
        "--geometry-features",
        action="store_true",
        help="Include invariant chain and mutation-distance context features",
    )
    parser.add_argument("--delta-loss", choices=("mse", "smooth_l1"), default="mse")
    parser.add_argument("--delta-loss-beta", type=float, default=1.0)
    parser.add_argument(
        "--mutation-loss-weight",
        type=float,
        default=0.0,
        help="Extra weight for mutated residues in the supervised loss",
    )
    parser.add_argument(
        "--neighborhood-loss-weight",
        type=float,
        default=0.0,
        help="Extra weight for residues within 10 Angstrom of a mutation",
    )
    parser.add_argument(
        "--family-balanced-loss",
        action="store_true",
        help="Give each training family equal total supervised weight",
    )
    parser.add_argument(
        "--biochemical-edit-features",
        action="store_true",
        help="Append fixed biochemical target-minus-source descriptors",
    )
    parser.add_argument(
        "--ablate-target-residue",
        action="store_true",
        help="Zero target-residue one-hot channels while retaining parent identity and mutation position",
    )
    parser.add_argument(
        "--neighborhood-radius",
        type=float,
        default=10.0,
        help="Radius in Angstrom used by neighborhood loss weighting",
    )
    parser.add_argument(
        "--teacher-cache",
        help="Admitted teacher cache index.json for optional delta distillation",
    )
    parser.add_argument(
        "--teacher-noise-level",
        type=float,
        help="Noise level to read from --teacher-cache",
    )
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=0.0,
        help="Weight of the optional teacher-delta loss",
    )
    parser.add_argument(
        "--max-normalized-delta",
        type=float,
        default=None,
        help="Optional tanh bound for each predicted normalized delta channel",
    )
    parser.add_argument("--no-positional-encoding", action="store_true")
    parser.add_argument("--no-gradient-clip", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--allow-nonexperimental", action="store_true")
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()
    if args.ablate_target_residue and args.biochemical_edit_features:
        parser.error("--ablate-target-residue cannot be combined with --biochemical-edit-features")

    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise SystemExit("ospedit-train requires torch; install ospedit[torch]") from error
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = load_manifest(args.manifest)
    errors = validate_manifest(records)
    if args.verify_checksums:
        errors.extend(error for record in records for error in verify_record_checksums(record))
    if errors:
        raise SystemExit("manifest audit failed: " + "; ".join(errors))
    selected = [record for record in records if record.split == args.split and (args.allow_nonexperimental or record.label_source == "experimental")]
    if not selected:
        raise SystemExit(f"manifest contains no records for split={args.split!r}")
    teacher_cache = None
    if args.teacher_cache:
        try:
            teacher_cache = TeacherCache.load(args.teacher_cache, records=records, split=args.split)
        except (FileNotFoundError, KeyError, ValueError) as error:
            raise SystemExit(f"teacher cache validation failed: {error}") from error
        if args.teacher_noise_level is None:
            raise SystemExit("--teacher-noise-level is required with --teacher-cache")
    elif args.teacher_noise_level is not None:
        raise SystemExit("--teacher-noise-level requires --teacher-cache")
    if args.distill_weight and teacher_cache is None:
        raise SystemExit("--distill-weight requires --teacher-cache")
    teacher_cache_fingerprint = teacher_cache.metadata.get("manifest_fingerprint") if teacher_cache is not None else None
    parent_dim = int(parent_local_features(selected[0].pair, include_geometry=args.geometry_features).shape[-1])
    edit_dim = 48 if args.biochemical_edit_features else 41
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_config = dict(resume_payload.get("config", {}))
        expected_fingerprint = resume_config.get("manifest_fingerprint")
        if expected_fingerprint and expected_fingerprint != manifest_fingerprint(records):
            raise SystemExit("resume checkpoint manifest fingerprint does not match --manifest")
    else:
        resume_config = {}
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else int(resume_config.get("hidden_dim", 256))
    blocks = args.blocks if args.blocks is not None else int(resume_config.get("blocks", 4))
    heads = args.heads if args.heads is not None else int(resume_config.get("heads", 8))
    max_normalized_delta = args.max_normalized_delta if args.max_normalized_delta is not None else resume_config.get("max_normalized_delta")
    use_positional_encoding = not args.no_positional_encoding
    if args.resume and "no_positional_encoding" not in resume_config:
        use_positional_encoding = False
    architecture = str(resume_config.get("student_architecture", args.student_architecture)) if args.resume else args.student_architecture
    if args.resume and architecture != args.student_architecture:
        raise SystemExit(f"resume checkpoint student_architecture={architecture} does not match requested {args.student_architecture}")
    model: Any
    if architecture == "spatial_graph_global":
        model = HybridSpatialGraphStudent(
            parent_dim=parent_dim,
            edit_dim=edit_dim,
            hidden_dim=hidden_dim,
            graph_blocks=blocks,
            global_blocks=args.global_blocks,
            heads=heads,
            max_normalized_delta=max_normalized_delta,
        )
    elif architecture == "spatial_graph":
        model = SpatialGraphStudent(
            parent_dim=parent_dim,
            edit_dim=edit_dim,
            hidden_dim=hidden_dim,
            blocks=blocks,
            max_normalized_delta=max_normalized_delta,
        )
    else:
        model = ParentEditStudent(
            parent_dim=parent_dim,
            edit_dim=edit_dim,
            hidden_dim=hidden_dim,
            blocks=blocks,
            heads=heads,
            max_normalized_delta=max_normalized_delta,
            use_positional_encoding=use_positional_encoding,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    prior_history: list[float] = []
    if args.resume:
        if resume_payload is None:  # defensive guard for type checkers
            raise SystemExit("resume checkpoint payload was not loaded")
        validate_student_checkpoint_config(
            dict(resume_payload.get("config", {})),
            parent_dim=parent_dim,
            edit_dim=edit_dim,
            hidden_dim=hidden_dim,
            blocks=blocks,
            heads=heads,
        )
        saved_loss_schema = resume_config.get("loss_schema")
        if saved_loss_schema not in {None, LOSS_SCHEMA_VERSION}:
            raise SystemExit(f"unsupported resume loss schema: {saved_loss_schema}")
        if saved_loss_schema is None and any(float(resume_config.get(name, 0.0)) > 0 for name in ("mutation_loss_weight", "neighborhood_loss_weight")):
            raise SystemExit("cannot resume a region-weighted checkpoint without a loss_schema")
        for key, requested in (
            ("translation_scale", args.translation_scale),
            ("rotation_scale", args.rotation_scale),
        ):
            saved = resume_config.get(key)
            if saved is not None and not np.isclose(float(saved), requested):
                raise SystemExit(f"resume checkpoint {key}={saved} does not match requested {requested}")
        saved_geometry = bool(resume_config.get("geometry_features", False))
        if saved_geometry != args.geometry_features:
            raise SystemExit(f"resume checkpoint geometry_features={saved_geometry} does not match requested {args.geometry_features}")
        saved_biochemical = bool(resume_config.get("biochemical_edit_features", False))
        if saved_biochemical != args.biochemical_edit_features:
            raise SystemExit(f"resume checkpoint biochemical_edit_features={saved_biochemical} does not match requested {args.biochemical_edit_features}")
        saved_target_ablation = bool(resume_config.get("ablate_target_residue", False))
        if saved_target_ablation != args.ablate_target_residue:
            raise SystemExit(
                f"resume checkpoint ablate_target_residue={saved_target_ablation} does not match requested {args.ablate_target_residue}"
            )
        saved_radius = resume_config.get("neighborhood_radius")
        if saved_radius is not None and not np.isclose(float(saved_radius), args.neighborhood_radius):
            raise SystemExit(f"resume checkpoint neighborhood_radius={saved_radius} does not match requested {args.neighborhood_radius}")
        saved_spatial_neighbors = resume_config.get("spatial_neighbors")
        if saved_spatial_neighbors is not None and int(saved_spatial_neighbors) != args.spatial_neighbors:
            raise SystemExit(f"resume checkpoint spatial_neighbors={saved_spatial_neighbors} does not match requested {args.spatial_neighbors}")
        saved_global_blocks = resume_config.get("global_blocks")
        if saved_global_blocks is not None and int(saved_global_blocks) != args.global_blocks:
            raise SystemExit(f"resume checkpoint global_blocks={saved_global_blocks} does not match requested {args.global_blocks}")
        saved_family_balanced = bool(resume_config.get("family_balanced_loss", False))
        if saved_family_balanced != args.family_balanced_loss:
            raise SystemExit(f"resume checkpoint family_balanced_loss={saved_family_balanced} does not match requested {args.family_balanced_loss}")
        for key, requested in (
            ("delta_norm_weight", args.delta_norm_weight),
            ("mutation_loss_weight", args.mutation_loss_weight),
            ("neighborhood_loss_weight", args.neighborhood_loss_weight),
            ("delta_loss_beta", args.delta_loss_beta),
        ):
            saved = resume_config.get(key)
            if saved is not None and not np.isclose(float(saved), requested):
                raise SystemExit(f"resume checkpoint {key}={saved} does not match requested {requested}")
        saved_loss = resume_config.get("delta_loss")
        if saved_loss is not None and saved_loss != args.delta_loss:
            raise SystemExit(f"resume checkpoint delta_loss={saved_loss} does not match requested {args.delta_loss}")
        saved_bound = resume_config.get("max_normalized_delta")
        if args.max_normalized_delta is not None:
            if saved_bound is None or not np.isclose(float(saved_bound), args.max_normalized_delta):
                raise SystemExit(f"resume checkpoint max_normalized_delta={saved_bound} does not match requested {args.max_normalized_delta}")
        saved_teacher_fingerprint = resume_config.get("teacher_cache_fingerprint")
        if saved_teacher_fingerprint is not None and saved_teacher_fingerprint != teacher_cache_fingerprint:
            raise SystemExit("resume checkpoint teacher cache fingerprint does not match requested cache")
        saved_teacher_level = resume_config.get("teacher_noise_level")
        if saved_teacher_level is not None:
            if args.teacher_noise_level is None or not np.isclose(float(saved_teacher_level), args.teacher_noise_level):
                raise SystemExit("resume checkpoint teacher noise level does not match requested value")
        saved_distill_weight = resume_config.get("distill_weight")
        if saved_distill_weight is not None and not np.isclose(float(saved_distill_weight), args.distill_weight):
            raise SystemExit("resume checkpoint distill weight does not match requested value")
        metadata = load_student_checkpoint(model, args.resume, optimizer=optimizer, map_location=args.device)
        start_epoch = metadata["epoch"]
        prior_history = metadata["history"]
    history = train_records(
        model,
        selected,
        optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        device=args.device,
        grad_accumulation_steps=args.grad_accumulation_steps,
        gradient_clip_norm=None if args.no_gradient_clip else args.gradient_clip_norm,
        translation_scale=args.translation_scale,
        rotation_scale=args.rotation_scale,
        delta_norm_weight=args.delta_norm_weight,
        include_geometry=args.geometry_features,
        delta_loss_kind=args.delta_loss,
        delta_loss_beta=args.delta_loss_beta,
        mutation_loss_weight=args.mutation_loss_weight,
        neighborhood_loss_weight=args.neighborhood_loss_weight,
        neighborhood_radius=args.neighborhood_radius,
        distill_weight=args.distill_weight,
        teacher_cache=teacher_cache,
        teacher_noise_level=args.teacher_noise_level,
        include_spatial_graph=architecture in {"spatial_graph", "spatial_graph_global"},
        spatial_neighbors=args.spatial_neighbors,
        family_balanced_loss=args.family_balanced_loss,
        include_biochemical=args.biochemical_edit_features,
        include_target_residue=not args.ablate_target_residue,
    )
    evaluation = None
    evaluation_payload: dict[str, object] | None
    if args.eval_split is not None:
        evaluation_records = [record for record in records if record.split == args.eval_split and (args.allow_nonexperimental or record.label_source == "experimental")]
        if not evaluation_records:
            raise SystemExit(f"manifest contains no records for eval split={args.eval_split!r}")
        evaluation = evaluate_manifest_batched(
            evaluation_records,
            StudentEditor(
                model,
                device=args.device,
                translation_scale=args.translation_scale,
                rotation_scale=args.rotation_scale,
                include_geometry=args.geometry_features,
                include_spatial_graph=architecture in {"spatial_graph", "spatial_graph_global"},
                spatial_neighbors=args.spatial_neighbors,
                include_biochemical=args.biochemical_edit_features,
                include_target_residue=not args.ablate_target_residue,
            ),
            batch_size=args.eval_batch_size or args.batch_size,
            method="student",
            split=args.eval_split,
        )
        evaluation_payload = {
            "method": evaluation.method,
            "split": evaluation.split,
            "family_summary": evaluation.family_summary,
            "runtime_summary": evaluation.runtime_summary,
            "manifest_fingerprint": evaluation.manifest_fingerprint,
        }
        copy_evaluation = evaluate_manifest_batched(
            evaluation_records,
            CopyParentEditor(),
            batch_size=args.eval_batch_size or args.batch_size,
            method="copy_parent_baseline",
            split=args.eval_split,
        )
        evaluation_payload["copy_parent_baseline"] = {
            "method": copy_evaluation.method,
            "split": copy_evaluation.split,
            "family_summary": copy_evaluation.family_summary,
            "runtime_summary": copy_evaluation.runtime_summary,
            "manifest_fingerprint": copy_evaluation.manifest_fingerprint,
        }
    else:
        evaluation_payload = None
    save_student_checkpoint(
        model,
        args.output,
        optimizer=optimizer,
        epoch=start_epoch + args.epochs,
        history=prior_history + history,
        config=vars(args)
        | {
            "loss_schema": LOSS_SCHEMA_VERSION,
            "parent_dim": parent_dim,
            "edit_dim": edit_dim,
            "hidden_dim": hidden_dim,
            "blocks": blocks,
            "heads": heads,
            "record_count": len(selected),
            "manifest_fingerprint": manifest_fingerprint(records),
            "teacher_cache_fingerprint": teacher_cache_fingerprint,
            "evaluation": evaluation_payload,
        },
    )
    print(
        json.dumps(
            json_safe(
                {
                    "output": args.output,
                    "records": len(selected),
                    "epochs": start_epoch + args.epochs,
                    "final_loss": history[-1],
                    "evaluation": evaluation_payload,
                }
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
