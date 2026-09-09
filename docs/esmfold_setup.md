# ESMFold baseline integration

This integration evaluates sequence-only refolding against reference-based
editors. Both methods use the same residue mapping and metric implementation;
ESMFold predictions receive one global backbone Kabsch alignment to the parent
before coordinate-frame metrics are computed.

## Modes

`B2_esmfold_default` passes `num_recycles=None`, selecting the checkpoint's
training default. `B2_esmfold_zero_extra_recycles` passes `num_recycles=0`.
In fair-esm 2.0.0, the folding trunk increments an explicit value by one, so
zero means one standard trunk/structure-module pass and no additional recycle.
It does not mean that the folding trunk is skipped.

The runner loads one resident model and evaluates all requested modes and
sequences. Model-load time is reported separately and divided equally across
new jobs for the end-to-end amortized value. Pure inference time and peak VRAM
are also retained per prediction. Reading an existing prediction cache does
not replace these recorded generation costs.

## Tested environment

- environment: `/home/husrcf/anaconda3/envs/BIO`
- PyTorch: `2.9.1+rocm7.2.0.git7e1940d4`
- fair-esm: `2.0.0`
- device: AMD Radeon AI PRO R9700
- ESM precision: bfloat16
- chunk size: 64

The local ESMFold checkpoint contains compatibility aliases for the IPA point
projection keys expected by fair-esm 2.0.0. Its identity is therefore recorded
by checksum rather than assumed to match an unmodified download:

| file | SHA-256 |
| --- | --- |
| `esmfold_3B_v1.pt` | `569c07a3b62c70406787555aa2249d95d99acc10093df4d42d26aa8ae38af61c` |
| `esm2_t36_3B_UR50D.pt` | `7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500` |
| contact regression | `4da500eab246481dc9c8c95bc7b1d02f2803d761c380b0e95186d4a07d0fc84e` |

## Command

```bash
python -m scripts.run_esmfold_baseline \
  --manifest data/manifest/pairs.jsonl \
  --split test \
  --model-dir /path/to/torch-hub-cache \
  --python /path/to/esmfold-environment/bin/python \
  --prediction-dir results/esmfold_predictions \
  --runner examples/esmfold_predict.py \
  --chunk-size 64 --esm-precision bf16 \
  --verify-checksums \
  --output results/esmfold_test.json \
  --csv-output results/esmfold_test.csv
```

The model directory must contain the three files listed above under
`checkpoints/`. Cache identity includes their checksums, the runner checksum,
sequence hash, recycle mode, precision, and chunk size.

## Frozen three-pair result

| split | method | local error | site error | distance error | distance cosine | predicted / true norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | copy | 0.116 | 0.093 | 0.343 | undefined | 0 / 52.3 |
| train | default | 0.274 | 0.247 | 0.580 | 0.213 | 83.4 / 52.3 |
| train | zero extra | 0.315 | 0.302 | 0.625 | 0.173 | 89.3 / 52.3 |
| dev | copy | 0.357 | 0.175 | 0.359 | undefined | 0 / 44.7 |
| dev | default | 0.400 | 0.319 | 0.713 | 0.152 | 83.8 / 44.7 |
| dev | zero extra | 0.928 | 0.501 | 0.860 | 0.049 | 99.6 / 44.7 |
| test | copy | 0.718 | 0.792 | 0.362 | undefined | 0 / 41.0 |
| test | default | 1.077 | 1.020 | 0.875 | 0.082 | 93.9 / 41.0 |
| test | zero extra | 1.025 | 1.055 | 0.847 | 0.165 | 93.9 / 41.0 |

Errors are Angstrom-valued ospedit metrics except cosine and distance-change
norm. Neither refolding mode beats copy-parent on local, site, or distance
error in this minimum set. Their nonzero direction cosine does not compensate
for excessive response magnitude and remote scaffold drift of `0.44--0.63
Angstrom`.

Pure GPU inference took `3.75--4.15 s` for default recycling and `0.31--0.41
s` for zero extra recycling at lengths 114--153. Peak allocated VRAM was about
`8.6 GB`. Shared model loading took about 25 seconds per two-job split run and
is reported separately; larger persistent batches are required for a stable
throughput comparison.

This is a three-record mechanism comparison, not a general ESMFold benchmark.
