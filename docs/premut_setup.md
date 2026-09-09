# PreMut baseline integration

This integration evaluates the published raw PreMut checkpoint under the same
backbone mapping and metrics as ospedit. It does not run the optional
ATOMRefine stage.

## Compatibility boundary

PreMut is executed in a separate Python process because its published
environment targets PyTorch 1.12 and PyG 2.2. The tested local checkout is
commit `f0ec72963e247f897b6163d01b5c5920e96fca80`; the published checkpoint has
SHA-256
`3d72056f4d06278270bfcf912539c6b7c0848ddc580a256fc0dd906f9efee792`.

The adapter requires exactly one substitution, one chain, consecutive author
residue numbers, and no insertion codes. This reflects PreMut's own
zero-based-to-author-number conversion. It rejects an input if the backbone
reconstructed by PreMut differs from the manifest parent by more than
`0.002 Angstrom` RMSD.

The upstream parser centers all atoms before inference and initializes missing
mutant side-chain atoms randomly. `examples/premut_predict.py` therefore:

- fixes Python, NumPy, and Torch seeds;
- restores the original parent translation after inference;
- extracts N, CA, C, and O in residue order;
- saves the reconstructed input backbone for an independent frame check;
- records source, checkpoint, runner, and upstream revision identities.

Changing any cache identity causes a fresh prediction. A source checksum
mismatch fails before cache reuse.

## Environment

Use a dedicated environment containing the dependencies from PreMut's
`environment.yml`. The local smoke used the existing `dock` environment plus:

```bash
conda run -n dock pip install biopandas==0.4.1
```

This is a compatibility smoke on a newer Torch/PyG stack, not a claim of exact
numerical equivalence to the authors' original CUDA 11.3 environment.

## Evaluation

```bash
python scripts/run_premut_baseline.py \
  --manifest data/manifest/pairs.jsonl \
  --split test \
  --upstream-root /path/to/PreMut \
  --checkpoint /path/to/PreMut/Saved_Model/model.ckpt \
  --python /path/to/premut-environment/bin/python \
  --prediction-dir results/premut_predictions_seed0 \
  --runner examples/premut_predict.py \
  --device cpu --seed 0 --verify-checksums \
  --output results/premut_test_seed0.json \
  --csv-output results/premut_test_seed0.csv
```

The report always includes `C0_copy_parent` and `B1_premut_raw`. A fresh
run uses `--execution-mode persistent_batch` by default: one external process
loads PreMut once, predicts all missing records, and records model loading,
preprocessing, inference, and postprocessing separately. Use `--split all` to
measure a whole frozen manifest in one resident process. The generated cache
then replays those generation costs into the ospedit suite report; a reported
cache handoff is not interpreted as zero-cost inference.

`--execution-mode cold_subprocess` remains available for compatibility and
runs one process per missing pair. A pre-existing cache without a matching
generation-cost report measures retrieval only and must not be presented as
warm-model inference speed.

On the frozen three-pair set, the persistent CPU run loaded the model in
`2.204 s`. Candidate work totaled `5.028 s` (`4.939 s` preprocessing,
`0.087 s` network inference, and `0.001 s` postprocessing), for an observed
resident total of `7.232 s`. The immediate optimization target is therefore
the repeated PDB/30-neighbor graph preprocessing, especially for many mutants
of one parent.

The same persistent protocol was measured on 15 experimental mutations of
the 164-residue parent `2LZM_A`. Before parent parsing reuse, model loading took
`2.185 s`; candidate work averaged `2.039 s`, for an observed N=15 resident
total of `32.775 s`.

The runner now caches the identity-filtered parent ATOM table across jobs. It
does not cache mutation-dependent graphs and deliberately preserves PreMut's
second randomized coordinate construction for KNN edges. All 15 predictions
and reconstructed inputs were elementwise identical before and after this
change. The equivalent cached run took `2.183 s` to load and `1.892 s` per
candidate, or `30.556 s` total. Preprocessing still dominated (`27.790 s`)
over network inference (`0.575 s`), so an exact, vectorized implementation of
the mutation-dependent graph builder is the next plausible optimization.

## Frozen three-pair result

The multi-repeat-identifiable mechanism set contains one record in each split:
H63T (train), T49V (dev), and D31A (test). The following values are from seed
0 without refinement.

| split | method | local error | mutation-site error | distance-change error | distance cosine | predicted / true change norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | copy | 0.11590 | 0.09329 | 0.34306 | undefined | 0 / 52.32 |
| train | PreMut | 0.11666 | 0.09480 | 0.34300 | 0.028 | 2.53 / 52.32 |
| dev | copy | 0.35718 | 0.17492 | 0.35885 | undefined | 0 / 44.68 |
| dev | PreMut | 0.36003 | 0.17524 | 0.36051 | -0.066 | 2.27 / 44.68 |
| test | copy | 0.71790 | 0.79203 | 0.36165 | undefined | 0 / 41.05 |
| test | PreMut | 0.71757 | 0.78225 | 0.36551 | -0.098 | 3.21 / 41.05 |

All errors are Angstrom-valued ospedit metrics except cosine and the distance
change norms. PreMut slightly improves the D31A local and mutation-site errors
but worsens its distance-change error and points in the wrong aggregate
distance-change direction. It does not beat copy-parent consistently on this
three-record mechanism set.

Seeds 0, 1, and 2 were also run. D31A is exactly seed-invariant because the
mutation introduces no missing atom initialized by PreMut. H63T and T49V show
small but nonzero sensitivity: mutation-site error ranges are `0.00134` and
`0.00021 Angstrom`, and predicted distance-change norm ranges are `0.0074`
and `0.0645`. Future PreMut comparisons must predeclare seeds and aggregate
them rather than select one favorable initialization.

These three records are a mechanism check, not a PreMut benchmark. They are
too few for a performance claim, and the checkpoint's exposure to individual
MutData2022 structures has not been ruled out.
