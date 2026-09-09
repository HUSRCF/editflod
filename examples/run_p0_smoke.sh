#!/usr/bin/env bash
set -euo pipefail

output_dir="$(mktemp -d "${TMPDIR:-/tmp}/ospedit-p0.XXXXXX")"
trap 'rm -rf "$output_dir"' EXIT

ospedit-eval \
  --parent-structure "$(dirname "$0")/toy_parent.pdb" \
  --mutant-structure "$(dirname "$0")/toy_mutant.pdb" \
  --parent-chain A \
  --mutant-chain A \
  --pair-id toy_pdb \
  --manifest-output "$output_dir/manifest.jsonl"

ospedit-eval --audit-manifest "$output_dir/manifest.jsonl"
ospedit-eval --manifest "$output_dir/manifest.jsonl" --eval-split dev >/dev/null
printf 'P0 smoke passed; temporary outputs cleaned\n'
