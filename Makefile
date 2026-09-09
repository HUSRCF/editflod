.PHONY: test lint typecheck compile package smoke mechanism-smoke check

test:
	python -m pytest -q

lint:
	ruff check scripts/build_protocol_v02_manifest.py
	ruff check scripts/export_student_checkpoint_report.py
	ruff check scripts/audit_response_learnability.py
	ruff check scripts/summarize_student_replicates.py
	ruff check src/ospedit/data.py src/ospedit/adapters.py src/ospedit/external_baselines.py src/ospedit/foldflow_batch.py src/ospedit/foldflow_env.py src/ospedit/models.py src/ospedit/experiment.py src/ospedit/oracle.py src/ospedit/manifest_cli.py src/ospedit/split_cli.py src/ospedit/p1_preflight.py src/ospedit/teacher_evaluation.py scripts/run_oracle_reconstruction.py scripts/run_mechanism_grid.py scripts/run_mechanism_sweep.py scripts/select_mechanism_config.py scripts/run_teacher_admission.py scripts/build_teacher_cache.py scripts/evaluate_teacher_cache.py scripts/evaluate_teacher_combinations.py scripts/audit_microminer_group_targets.py scripts/audit_microminer_metadata.py scripts/build_microminer_manifest.py scripts/build_premut_manifest.py scripts/build_platinum_manifest.py scripts/discover_rcsb_repeats.py scripts/select_microminer_candidates.py scripts/select_microminer_repeat_groups.py scripts/select_premut_repeat_candidates.py scripts/merge_manifests.py scripts/audit_response_scale.py scripts/audit_repeat_structures.py scripts/audit_structure_context.py scripts/run_student_bound_sweep.py scripts/run_premut_baseline.py scripts/run_esmfold_baseline.py scripts/assemble_benchmark_table.py scripts/assemble_runtime_table.py examples/foldflow_endpoint_factory.py examples/premut_predict.py examples/premut_batch_predict.py examples/esmfold_predict.py

typecheck:
	mypy scripts/build_protocol_v02_manifest.py --ignore-missing-imports
	mypy scripts/export_student_checkpoint_report.py --ignore-missing-imports
	mypy scripts/audit_response_learnability.py --ignore-missing-imports
	mypy scripts/summarize_student_replicates.py --ignore-missing-imports
	mypy src/ospedit/data.py src/ospedit/foldflow_env.py src/ospedit/foldflow_batch.py src/ospedit/adapters.py src/ospedit/external_baselines.py src/ospedit/models.py src/ospedit/experiment.py src/ospedit/oracle.py src/ospedit/manifest_cli.py src/ospedit/split_cli.py src/ospedit/p1_preflight.py src/ospedit/teacher_cache.py src/ospedit/teacher_evaluation.py src/ospedit/student_data.py src/ospedit/student_training.py src/ospedit/train_cli.py scripts/run_oracle_reconstruction.py scripts/run_student_bound_sweep.py scripts/run_mechanism_sweep.py scripts/select_mechanism_config.py scripts/evaluate_teacher_cache.py scripts/evaluate_teacher_combinations.py scripts/audit_response_scale.py scripts/audit_repeat_structures.py scripts/audit_structure_context.py scripts/audit_microminer_group_targets.py scripts/audit_microminer_metadata.py scripts/build_microminer_manifest.py scripts/build_premut_manifest.py scripts/build_platinum_manifest.py scripts/discover_rcsb_repeats.py scripts/select_microminer_candidates.py scripts/select_microminer_repeat_groups.py scripts/select_premut_repeat_candidates.py scripts/merge_manifests.py scripts/run_premut_baseline.py scripts/run_esmfold_baseline.py scripts/assemble_benchmark_table.py scripts/assemble_runtime_table.py --ignore-missing-imports

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

check: test lint typecheck compile package smoke mechanism-smoke
