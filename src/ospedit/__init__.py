"""Minimal research framework for mutation-conditioned structure editing."""

from .data import PairRecord, StructurePair, append_manifest, assign_group_splits, file_sha256, json_safe, load_manifest, manifest_fingerprint, pair_record_from_structures, pair_parsed_structures, parse_structure, validate_manifest, verify_record_checksums, write_manifest
from .experiment import EvaluationResult, ManifestEvaluation, SuiteEvaluation, build_mechanism_editors, evaluate_editor, evaluate_editor_suite, evaluate_manifest, evaluate_manifest_batched, evaluate_parent_workloads, flatten_parent_workload_reports, flatten_suite_reports, group_records_by_parent, parent_workloads, parent_workload_payload, suite_payload, write_parent_workload_csv, write_parent_workload_report, write_suite_csv, write_suite_report
from .models import ConditionalDifferenceEditor, CopyParentEditor, FieldModel, IndependentNoiseLocalFrameDifferenceEditor, LocalFrameDifferenceEditor, MultiNoiseLocalFrameDifferenceEditor, MutationNeighborhoodDifferenceEditor, RepeatedSingleNoiseLocalFrameDifferenceEditor, StudentEditor, TargetUpdateEditor
from .metrics import evaluate_pair
from .diagnostics import assess_teacher_admission, condition_response_diagnostic, condition_response_repeat_error
from .noise import SharedNoise, make_shared_noise
from .student import ParentEditStudent, encode_edit_features
from .student_data import PairDataset, collate_pair_records, iter_pair_batches, parent_local_features, parent_residue_mask, target_local_delta
from .student_training import load_student_checkpoint, masked_delta_loss, masked_prediction_norm_loss, save_student_checkpoint, train_records, train_student, validate_student_checkpoint_config
from .student_inference import ParentContextCache, apply_student_delta, predict_student, predict_student_batch
from .oracle import oracle_local_delta, oracle_prediction
from .adapters import FoldFlow2EndpointAdapter
from .external_baselines import ESMFoldEditor, PreMutEditor
from .foldflow_batch import FoldFlow2BatchBuilder, FoldFlow2MarginalConverter, backbone_to_rigids, make_foldflow_reference_rigid_sampler, make_forward_marginal_sampler, openfold_rigid_from_tensor7, rotation_matrix_to_quaternion, sequence_to_aatype
from .foldflow_env import inspect_foldflow_environment
from .p1_preflight import p1_preflight_report
from .teacher_cache import TeacherCache
from .teacher_evaluation import evaluate_teacher_cache

__all__ = [
    "StructurePair",
    "PairRecord",
    "load_manifest",
    "validate_manifest",
    "parse_structure",
    "pair_parsed_structures",
    "pair_record_from_structures",
    "append_manifest",
    "assign_group_splits",
    "manifest_fingerprint",
    "write_manifest",
    "file_sha256",
    "json_safe",
    "verify_record_checksums",
    "FieldModel",
    "CopyParentEditor",
    "TargetUpdateEditor",
    "ConditionalDifferenceEditor",
    "evaluate_pair",
    "condition_response_diagnostic",
    "condition_response_repeat_error",
    "assess_teacher_admission",
    "SharedNoise",
    "make_shared_noise",
    "ParentEditStudent",
    "encode_edit_features",
    "PairDataset",
    "collate_pair_records",
    "parent_local_features",
    "parent_residue_mask",
    "target_local_delta",
    "iter_pair_batches",
    "masked_delta_loss",
    "masked_prediction_norm_loss",
    "train_student",
    "train_records",
    "save_student_checkpoint",
    "load_student_checkpoint",
    "validate_student_checkpoint_config",
    "apply_student_delta",
    "predict_student",
    "predict_student_batch",
    "ParentContextCache",
    "oracle_local_delta",
    "oracle_prediction",
    "EvaluationResult",
    "evaluate_editor",
    "ManifestEvaluation",
    "evaluate_manifest",
    "evaluate_manifest_batched",
    "SuiteEvaluation",
    "evaluate_editor_suite",
    "build_mechanism_editors",
    "suite_payload",
    "write_suite_report",
    "flatten_suite_reports",
    "write_suite_csv",
    "group_records_by_parent",
    "parent_workloads",
    "evaluate_parent_workloads",
    "parent_workload_payload",
    "write_parent_workload_report",
    "flatten_parent_workload_reports",
    "write_parent_workload_csv",
    "LocalFrameDifferenceEditor",
    "MutationNeighborhoodDifferenceEditor",
    "MultiNoiseLocalFrameDifferenceEditor",
    "IndependentNoiseLocalFrameDifferenceEditor",
    "RepeatedSingleNoiseLocalFrameDifferenceEditor",
    "StudentEditor",
    "FoldFlow2EndpointAdapter",
    "PreMutEditor",
    "ESMFoldEditor",
    "FoldFlow2BatchBuilder",
    "FoldFlow2MarginalConverter",
    "make_foldflow_reference_rigid_sampler",
    "sequence_to_aatype",
    "backbone_to_rigids",
    "rotation_matrix_to_quaternion",
    "make_forward_marginal_sampler",
    "openfold_rigid_from_tensor7",
    "inspect_foldflow_environment",
    "p1_preflight_report",
    "TeacherCache",
    "evaluate_teacher_cache",
]
