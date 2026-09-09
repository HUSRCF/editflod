.PHONY: test lint typecheck compile package smoke mechanism-smoke validate-reports check

test:
	python -m pytest -q

lint:
	ruff check scripts/build_protocol_v02_manifest.py
	ruff check scripts/export_student_checkpoint_report.py
	ruff check scripts/audit_response_learnability.py
	ruff check scripts/audit_background_coverage.py
	ruff check scripts/analyze_background_signal.py
	ruff check scripts/audit_repeat_control_context.py
	ruff check scripts/audit_rcsb_environment.py
	ruff check scripts/audit_rcsb_endpoints.py
	ruff check scripts/audit_endpoint_mutation_annotations.py
	ruff check scripts/audit_manifest_endpoint_groups.py
	ruff check scripts/build_endpoint_context_queue.py
	ruff check scripts/select_endpoint_context_candidates.py
	ruff check scripts/select_structure_prediction_layer.py
	ruff check scripts/validate_endpoint_context_decisions.py
	ruff check scripts/build_response_review_queue.py
	ruff check scripts/summarize_student_replicates.py
	ruff check scripts/run_student_architecture_sweep.py
	ruff check scripts/analyze_student_generalization.py
	ruff check scripts/build_within_family_probe.py
	ruff check scripts/augment_reverse_training.py
	ruff check src/ospedit/data.py src/ospedit/adapters.py src/ospedit/external_baselines.py src/ospedit/foldflow_batch.py src/ospedit/foldflow_env.py src/ospedit/models.py src/ospedit/experiment.py src/ospedit/oracle.py src/ospedit/manifest_cli.py src/ospedit/split_cli.py src/ospedit/p1_preflight.py src/ospedit/teacher_evaluation.py scripts/run_oracle_reconstruction.py scripts/run_mechanism_grid.py scripts/run_mechanism_sweep.py scripts/select_mechanism_config.py scripts/run_teacher_admission.py scripts/build_teacher_cache.py scripts/evaluate_teacher_cache.py scripts/evaluate_teacher_combinations.py scripts/audit_microminer_group_targets.py scripts/audit_microminer_metadata.py scripts/build_microminer_manifest.py scripts/build_premut_manifest.py scripts/build_platinum_manifest.py scripts/discover_rcsb_repeats.py scripts/select_microminer_candidates.py scripts/select_microminer_repeat_groups.py scripts/select_premut_repeat_candidates.py scripts/merge_manifests.py scripts/audit_response_scale.py scripts/audit_repeat_structures.py scripts/audit_structure_context.py scripts/run_student_bound_sweep.py scripts/run_premut_baseline.py scripts/run_esmfold_baseline.py scripts/assemble_benchmark_table.py scripts/assemble_runtime_table.py examples/foldflow_endpoint_factory.py examples/premut_predict.py examples/premut_batch_predict.py examples/esmfold_predict.py

typecheck:
	mypy scripts/build_protocol_v02_manifest.py --ignore-missing-imports
	mypy scripts/export_student_checkpoint_report.py --ignore-missing-imports
	mypy scripts/audit_response_learnability.py --ignore-missing-imports
	mypy scripts/audit_background_coverage.py --ignore-missing-imports
	mypy scripts/analyze_background_signal.py --ignore-missing-imports
	mypy scripts/audit_repeat_control_context.py --ignore-missing-imports
	mypy scripts/audit_rcsb_environment.py --ignore-missing-imports
	mypy scripts/audit_rcsb_endpoints.py --ignore-missing-imports
	mypy scripts/audit_endpoint_mutation_annotations.py --ignore-missing-imports
	mypy scripts/audit_manifest_endpoint_groups.py --ignore-missing-imports
	mypy scripts/build_endpoint_context_queue.py --ignore-missing-imports
	mypy scripts/select_endpoint_context_candidates.py --ignore-missing-imports
	mypy scripts/select_structure_prediction_layer.py --ignore-missing-imports
	mypy scripts/validate_endpoint_context_decisions.py --ignore-missing-imports
	mypy scripts/build_response_review_queue.py --ignore-missing-imports
	mypy scripts/summarize_student_replicates.py --ignore-missing-imports
	mypy scripts/run_student_architecture_sweep.py --ignore-missing-imports
	mypy scripts/analyze_student_generalization.py --ignore-missing-imports
	mypy scripts/build_within_family_probe.py --ignore-missing-imports
	mypy scripts/augment_reverse_training.py --ignore-missing-imports
	mypy src/ospedit/data.py src/ospedit/structure_context.py src/ospedit/rcsb_metadata.py src/ospedit/pdb_annotations.py src/ospedit/endpoint_groups.py src/ospedit/foldflow_env.py src/ospedit/foldflow_batch.py src/ospedit/adapters.py src/ospedit/external_baselines.py src/ospedit/models.py src/ospedit/experiment.py src/ospedit/oracle.py src/ospedit/manifest_cli.py src/ospedit/split_cli.py src/ospedit/p1_preflight.py src/ospedit/teacher_cache.py src/ospedit/teacher_evaluation.py src/ospedit/student_data.py src/ospedit/student_training.py src/ospedit/train_cli.py scripts/run_oracle_reconstruction.py scripts/run_student_bound_sweep.py scripts/run_mechanism_sweep.py scripts/select_mechanism_config.py scripts/evaluate_teacher_cache.py scripts/evaluate_teacher_combinations.py scripts/audit_response_scale.py scripts/audit_repeat_structures.py scripts/audit_structure_context.py scripts/audit_microminer_group_targets.py scripts/audit_microminer_metadata.py scripts/build_microminer_manifest.py scripts/build_premut_manifest.py scripts/build_platinum_manifest.py scripts/discover_rcsb_repeats.py scripts/select_microminer_candidates.py scripts/select_microminer_repeat_groups.py scripts/select_premut_repeat_candidates.py scripts/merge_manifests.py scripts/run_premut_baseline.py scripts/run_esmfold_baseline.py scripts/assemble_benchmark_table.py scripts/assemble_runtime_table.py --ignore-missing-imports

compile:
	python -m compileall -q src tests scripts examples

package:
	@wheel_dir="$$(mktemp -d "$${TMPDIR:-/tmp}/ospedit-wheel.XXXXXX")"; \
	trap 'rm -rf "$$wheel_dir"' EXIT; \
	python -m pip wheel . --no-deps --no-build-isolation -w "$$wheel_dir" >/dev/null; \
	WHEEL_DIR="$$wheel_dir" python -c 'import os; from pathlib import Path; from zipfile import ZipFile; wheel=next(Path(os.environ["WHEEL_DIR"]).glob("*.whl")); archive=ZipFile(wheel); names=set(archive.namelist()); required={"ospedit/data.py", "ospedit/experiment.py", "ospedit/adapters.py", "ospedit/external_baselines.py", "ospedit/foldflow_batch.py", "ospedit/foldflow_env.py", "ospedit/split_cli.py", "ospedit/p1_preflight.py"}; assert required <= names, (required - names); metadata=next(archive.read(name).decode() for name in names if name.endswith(".dist-info/entry_points.txt")); commands={"ospedit-eval", "ospedit-train", "ospedit-foldflow-check", "ospedit-p1-preflight", "ospedit-build-manifest", "ospedit-split-manifest"}; assert all(f"{command} =" in metadata for command in commands), metadata'

smoke:
	bash examples/run_p0_smoke.sh

mechanism-smoke:
	python examples/run_mechanism_smoke.py

validate-reports:
	python scripts/validate_endpoint_context_decisions.py reports/protocol_v02_endpoint_context_queue_v1.json reports/protocol_v02_endpoint_context_decisions_v1.json

check: test lint typecheck compile package smoke mechanism-smoke validate-reports
