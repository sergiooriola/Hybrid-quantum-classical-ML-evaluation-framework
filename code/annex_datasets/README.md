# Dataset-robustness annex (Case Study 2)

This folder re-runs the Case Study 2 imbalance experiment
(`../imbalance_experiments.ipynb`) on four geometrically different toy datasets,
to show whether the quantum-vs-classical verdict obtained on `make_moons`
**transfers** to other problems. It is meant to back a thesis annex.

**The experiment is identical to the main study** -- same grid (2 noise levels x
8 models x 7 ratios x 12 seeds = 1 344 runs/dataset), same models, same splits,
same hyperparameters, same analysis code and same statistical protocol
(Friedman + Wilcoxon + Holm). Only the base dataset changes. The per-dataset
notebooks are produced by *transforming* the original notebook (copying every
analysis cell verbatim and patching only the data source and output paths), so
the numbers are directly comparable. The 12-seed budget is what lets the
pairwise protocol certify model pairs.

The four datasets reproduce Appendix E.1 of Rodriguez-Grasa, Ban & Sanz,
*"Neural quantum kernels: training quantum kernels with quantum neural
networks"* (arXiv:2401.04642v2):

| name      | definition                                                        |
|-----------|-------------------------------------------------------------------|
| `sinus`   | points above / below `f(x1) = -0.8 sin(pi x1)`                    |
| `corners` | four quarter-circles of radius 0.75 at the corners of `[-1, 1]^2` |
| `spiral`  | two interleaved Archimedean spirals                               |
| `circles` | annular ring between radii `0.5*sqrt(2/pi)` and `sqrt(2/pi)`      |

## The dataset library

`../toy_datasets.py` provides the four generators with the **same interface as
`sklearn.datasets.make_moons`**, so they are drop-in replacements:

```python
import toy_datasets as td

X, y = td.make_sinus(n_samples=2000, noise=0.1, random_state=0)
X, y = td.make_corners(...)
X, y = td.make_spiral(...)
X, y = td.make_circles(...)        # the paper's annulus, NOT sklearn.make_circles

# or by name, plus a registry to iterate over:
X, y = td.make_dataset('spiral', n_samples=2000, noise=0.1, random_state=0)
for name in td.DATASETS:
    X, y = td.make_dataset(name)
```

`X` is `(n_samples, 2)`, `y` is in `{0, 1}` and **exactly 50/50 balanced** (so the
"balanced base, then artificially imbalanced" design of Case Study 2 is intact).
`noise` is the std of Gaussian jitter added to the coordinates, exactly like
`make_moons`.

## Files

| file                       | what it is                                                        |
|----------------------------|-------------------------------------------------------------------|
| `../toy_datasets.py`       | the dataset library (lives next to `qml_core`/`qml_imbalance`)    |
| `run_annex.py`             | full-grid parallel runner + cache loader (`CONFIG`, `load_records`) |
| `build_notebooks.py`       | regenerates the four `annex_<dataset>.ipynb` by transforming the original notebook |
| `build_summary.py`         | regenerates `summary.ipynb`                                       |
| `annex_<dataset>.ipynb`    | per-dataset replica of the Case Study 2 notebook (analysis byte-identical) |
| `summary.ipynb`            | the four datasets side by side (headline annex figures + advantage table) |
| `results/annex_*.json`     | cached run records (created by `run_annex.py`; same format as the main study) |
| `figures/`                 | figures saved by the notebooks (`<dataset>-case2-*.png`)          |

## How to run

1. Precompute the four caches once. The runner is parallel (joblib/loky); set
   the worker count with `N_JOBS` (default 12). On a 12-16 core machine the
   whole thing is ~25-40 min; sequential it is a few hours.

   ```
   N_JOBS=12 python run_annex.py        # all four
   N_JOBS=12 python run_annex.py spiral # just one
   ```

   It checkpoints every 24 tasks into `results/annex_<dataset>.json` and resumes
   from there if interrupted.

2. Open any `annex_<dataset>.ipynb` and "Run All". With the cache present the
   grid cell loads instantly (the heavy compute is already done) and the
   notebook just redraws the figures and re-runs the statistics. The two
   ablation cells (section 13) train a few small models inline (~1-2 min).
   `summary.ipynb` reads all four caches and produces the cross-dataset figures
   and the certified quantum-advantage table.

## Grid: identical to the main study

The grid is **identical** to `code/run_experiments.py` (see `CONFIG` in
`run_annex.py`), so the annex numbers are directly comparable to Case Study 2:

| knob          | main study and annex          |
|---------------|-------------------------------|
| noise levels  | 0.1, 0.3                      |
| ratios        | 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99 (7) |
| seeds         | 12                            |
| quantum iters | 200                           |
| models        | 3 quantum + 5 classical (8)   |
| runs/dataset  | 1 344                         |

**Why 12 seeds.** The pairwise Wilcoxon + Holm certification needs enough seeds
for its discrete-distribution floor to fall below 0.05: with `k = 8` models the
smallest Holm-adjusted p is `28 * 2/2^n`, equal to `0.0547` at `n = 10` (just
above 0.05) and `0.0137` at `n = 12`. The study fixes `n = 12` (matching the
main study's pre-registered choice), so the protocol can certify pairs -- which
is what makes the cross-dataset quantum-advantage table in `summary.ipynb`
meaningful.
