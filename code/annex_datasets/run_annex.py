"""
run_annex.py
============
Full-grid experiment runner for the *dataset-robustness annex* of Case Study 2.

It runs the **exact same experiment** as `code/run_experiments.py` /
`code/imbalance_experiments.ipynb` -- same grid, same models, same splits, same
hyperparameters -- but draws the balanced base pool from one of the four toy
datasets of Appendix E.1 of arXiv:2401.04642 (sinus, corners, spiral, circles)
via `toy_datasets.make_dataset` instead of `make_moons`. Keeping the grid
identical is what makes the annex numbers directly comparable to the main study,
and the 12-seed budget is what lets the pairwise Wilcoxon+Holm protocol certify
model pairs.

Grid per dataset: 2 noise x 8 models x 7 ratios x 12 seeds = 1 344 runs
(identical to the main study). Four datasets => 5 376 runs.

Parallelism. Tasks are independent ``run_one`` calls keyed on
(dataset, noise, model, ratio, seed), fanned out to ``joblib.Parallel`` (loky
process backend). Every random source inside ``run_one`` is keyed on
(seed, ratio) and the base pool is regenerated deterministically inside each
worker (``RandomState(base_seed)``), so results are bit-identical regardless of
worker count or completion order. Control the worker count with N_JOBS:

    N_JOBS=12 python run_annex.py            # all four datasets (default 12)
    N_JOBS=1  python run_annex.py spiral     # sequential, one dataset

Checkpointing. Per dataset, ``results/annex_<dataset>.json`` is written every
CHUNK_SIZE tasks and re-read on startup; already-computed
(noise, model, ratio, seed) tuples are skipped, so the run resumes after an
interruption. The cache format matches the main study's
``results/runs_simplified.json`` exactly, so the annex notebooks load it with
the same cell as the original.

Public API (used by summary.ipynb)
----------------------------------
    CONFIG                                 the (full, original) grid spec
    ALL_MODELS                             quantum + classical model names
    make_base(dataset, noise, config)      -> (X, y) balanced base pool
    load_records(dataset, results_dir)     -> list[dict]  (raises if no cache)
    records_to_results_by_noise(records)   -> {noise: list[qi.RunResult]}
"""

from __future__ import annotations

import os
import sys
import time
import json
from collections import defaultdict
from typing import Dict, List, Tuple

# Make the Case Study 2 library (one directory up) importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import numpy as np                       # noqa: E402
from joblib import Parallel, delayed     # noqa: E402
import qml_imbalance as qi               # noqa: E402
import toy_datasets as td                # noqa: E402


# =============================================================================
# Configuration -- identical to code/run_experiments.py CONFIG
# =============================================================================

CONFIG: Dict = dict(
    noise_levels     = [0.1, 0.3],
    n_total_base     = 8000,
    n_train          = 500,
    n_val            = 400,
    n_test           = 1200,
    ratios           = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99],
    seeds            = list(range(12)),
    quantum_iter     = 200,
    quantum_models   = ['data_reuploading', 'fixed_vqc', 'trainable_vqc'],
    classical_models = ['logreg', 'svm_rbf', 'knn', 'rf', 'mlp_small'],
    base_seed        = 42,
)

ALL_MODELS: List[str] = CONFIG['quantum_models'] + CONFIG['classical_models']

RESULTS_DIR = os.path.join(_HERE, 'results')
CHUNK_SIZE  = 24
N_JOBS      = int(os.environ.get('N_JOBS', '12'))


# =============================================================================
# Helpers
# =============================================================================

def make_base(dataset: str, noise: float, config: Dict = CONFIG
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Balanced base pool (X, y) for (dataset, noise), fixed across the grid.

    Mirrors the main study's ``make_balanced_dataset(n_total, noise, seed=42)``
    but for one of the toy datasets.
    """
    return td.make_dataset(dataset,
                           n_samples=config['n_total_base'],
                           noise=noise,
                           random_state=config['base_seed'])


def _model_factory(name: str, config: Dict):
    if name in qi.QuantumModel.SPEC:
        return qi.QuantumModel(name, n_iter=config['quantum_iter'])
    return qi.ClassicalModel(name)


def run_task(dataset: str, noise: float, mname: str, ratio: float, seed: int,
             config: Dict = CONFIG) -> Dict:
    """One (dataset, noise, model, ratio, seed) experiment -> serialisable dict.

    Top-level so joblib/loky can pickle it. Regenerates the base pool inside the
    worker (sub-millisecond) instead of shipping a 16k-float array per task.
    """
    X, y = make_base(dataset, noise, config)
    model = _model_factory(mname, config)
    r = qi.run_one(model, ratio=ratio, seed=seed, X_base=X, y_base=y,
                   n_train=config['n_train'], n_val=config['n_val'],
                   n_test=config['n_test'], verbose=False)
    return {
        'noise': noise, 'model': r.model, 'ratio': r.ratio, 'seed': r.seed,
        'metrics': r.metrics, 'threshold': r.threshold,
        'fit_time_s': r.fit_time_s, 'n_circuit_evals': r.n_circuit_evals,
        'n_train_majority': r.n_train_majority,
        'n_train_minority': r.n_train_minority,
    }


def _cache_path(dataset: str, results_dir: str = RESULTS_DIR) -> str:
    return os.path.join(results_dir, f'annex_{dataset}.json')


def load_records(dataset: str, results_dir: str = RESULTS_DIR) -> List[Dict]:
    """Load the cached run records for ``dataset`` (raises if absent)."""
    path = _cache_path(dataset, results_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'no cache at {path}; run `python run_annex.py {dataset}` first')
    with open(path) as f:
        return json.load(f)['runs']


def records_to_results_by_noise(records: List[Dict]
                                ) -> Dict[float, List[qi.RunResult]]:
    """Group cached dicts into {noise: [qi.RunResult]} (for aggregate/Friedman)."""
    out: Dict[float, List[qi.RunResult]] = defaultdict(list)
    for r in records:
        out[r['noise']].append(qi.RunResult(
            model=r['model'], ratio=r['ratio'], seed=r['seed'],
            metrics=r['metrics'], threshold=r['threshold'],
            fit_time_s=r['fit_time_s'], n_circuit_evals=r['n_circuit_evals'],
            n_train_majority=r['n_train_majority'],
            n_train_minority=r['n_train_minority']))
    return out


# =============================================================================
# Per-dataset runner (resume + parallel + checkpoint)
# =============================================================================

def run_dataset(dataset: str, config: Dict = CONFIG,
                results_dir: str = RESULTS_DIR, n_jobs: int = N_JOBS,
                verbose: bool = True) -> List[Dict]:
    path = _cache_path(dataset, results_dir)
    all_runs: List[Dict] = []
    if os.path.exists(path):
        with open(path) as f:
            all_runs = json.load(f).get('runs', [])
    done = {(r['noise'], r['model'], r['ratio'], r['seed']) for r in all_runs}

    fast_first = (['data_reuploading'] + config['classical_models']
                  + ['fixed_vqc', 'trainable_vqc'])
    tasks = [(noise, m, ratio, seed)
             for noise in config['noise_levels']
             for m in fast_first
             for ratio in config['ratios']
             for seed in config['seeds']
             if (noise, m, ratio, seed) not in done]

    if verbose:
        print(f'[{dataset}] cached={len(done)}  remaining={len(tasks)}  '
              f'N_JOBS={n_jobs}')
    if not tasks:
        if verbose:
            print(f'[{dataset}] already complete ({len(all_runs)} runs).')
        return all_runs

    def save():
        os.makedirs(results_dir, exist_ok=True)
        with open(path + '.tmp', 'w') as f:
            json.dump({'dataset': dataset, 'config': config, 'runs': all_runs},
                      f, indent=1, default=float)
        os.replace(path + '.tmp', path)

    pool = Parallel(n_jobs=n_jobs, backend='loky', batch_size=1)
    t0 = time.time()
    n_done = 0
    for start in range(0, len(tasks), CHUNK_SIZE):
        chunk = tasks[start:start + CHUNK_SIZE]
        results = pool(delayed(run_task)(dataset, *t, config) for t in chunk)
        all_runs.extend(results)
        save()
        n_done += len(results)
        if verbose:
            el = time.time() - t0
            eta = el / n_done * (len(tasks) - n_done) if n_done else 0
            last = results[-1]
            print(f'  [{dataset}] [{n_done:4d}/{len(tasks)}] '
                  f'noise={last["noise"]} {last["model"]:<18s} '
                  f'r={last["ratio"]:.2f} seed={last["seed"]:>2d}  '
                  f'BA={last["metrics"]["balanced_accuracy"]:.3f}  '
                  f'(elapsed {el/60:.1f}m, ETA {eta/60:.1f}m)')
    if verbose:
        print(f'[{dataset}] done: {len(all_runs)} runs in '
              f'{(time.time()-t0)/60:.1f} min -> {path}')
    return all_runs


def main(argv: List[str]) -> None:
    datasets = argv[1:] if len(argv) > 1 else list(td.DATASETS)
    bad = [d for d in datasets if d not in td.DATASETS]
    if bad:
        raise SystemExit(f'unknown dataset(s): {bad}; choose from {sorted(td.DATASETS)}')
    t0 = time.time()
    for d in datasets:
        run_dataset(d, verbose=True)
    print(f'\nAll done: {len(datasets)} dataset(s) in {(time.time()-t0)/60:.1f} min.')


if __name__ == '__main__':
    main(sys.argv)
