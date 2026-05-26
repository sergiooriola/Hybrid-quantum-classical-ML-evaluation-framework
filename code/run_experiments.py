"""
run_experiments.py
==================
Generate the experimental data for Case Study 2 of the thesis, with
joblib-parallel task execution and periodic JSON checkpointing.

Grid: 2 noise levels x 8 models x 7 ratios x 12 seeds = 1,344 runs.

Parallelism. Tasks are independent (each is a fully self-contained
``run_one`` call keyed on (model, ratio, seed)), so the driver fans them
out to ``joblib.Parallel`` workers using the loky process-based backend.
Process-based rather than threading because Python's GIL would otherwise
serialise the NumPy-bound forwards. Every random source inside
``run_one`` is keyed on (seed, ratio), so per-task results are
bit-identical regardless of n_jobs and regardless of completion order.

Set the number of workers with the N_JOBS environment variable:

    N_JOBS=1   python3 run_experiments.py   # sequential (default)
    N_JOBS=4   python3 run_experiments.py   # 4 workers
    N_JOBS=-1  python3 run_experiments.py   # joblib auto = all cores

Default is 1 to keep behaviour byte-identical to the reference single-
process run that produced results/runs_simplified.json shipped with the
project.

Checkpointing. The script writes results/runs_simplified.json after
every CHUNK_SIZE completed tasks. It also re-reads that file on startup
and skips any (noise, model, ratio, seed) tuples already present, so it
resumes after an interruption.
"""

import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from joblib import Parallel, delayed
import qml_imbalance as qi


# =============================================================================
# Configuration
# =============================================================================

N_TOTAL_BASE = 8000
N_TRAIN      = 500
N_VAL        = 400
N_TEST       = 1200
RATIOS       = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
NOISE_LEVELS = [0.1, 0.3]
ALL_SEEDS    = list(range(12))    # 12 seeds lifts Wilcoxon+Holm floor < alpha
QUANTUM_ITER = 200
QUANTUM_MODELS   = ['data_reuploading', 'fixed_vqc', 'trainable_vqc']
CLASSICAL_MODELS = ['logreg', 'svm_rbf', 'knn', 'rf', 'mlp_small']

OUT_PATH    = os.path.join(os.path.dirname(__file__), 'results', 'runs_simplified.json')
CHUNK_SIZE  = 8                                          # checkpoint cadence
N_JOBS      = int(os.environ.get('N_JOBS', '1'))         # see module docstring

CONFIG_OUT = {
    'noise_levels': NOISE_LEVELS,
    'n_total_base': N_TOTAL_BASE,
    'n_train': N_TRAIN, 'n_val': N_VAL, 'n_test': N_TEST,
    'ratios': RATIOS,
    'seeds': ALL_SEEDS,
    'quantum_iter': QUANTUM_ITER,
    'quantum_models': QUANTUM_MODELS,
    'classical_models': CLASSICAL_MODELS,
    'handler': 'class_weight',
    'tune_metric': 'balanced_accuracy',
}


# =============================================================================
# Worker function (top-level so joblib can pickle it for loky workers)
# =============================================================================

def run_task(noise, mname, ratio, seed,
             n_total=N_TOTAL_BASE, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
             quantum_iter=QUANTUM_ITER):
    """One (noise, model, ratio, seed) experiment. Returns a serialisable dict.

    Re-generates the base dataset inside the worker rather than receiving it
    as an argument: ``make_balanced_dataset`` is sub-millisecond and avoids
    repickling a 16k-float array on every job dispatch.
    """
    X, y = qi.make_balanced_dataset(n_total=n_total, noise=noise, seed=42)
    model = (qi.QuantumModel(mname, n_iter=quantum_iter)
             if mname in qi.QuantumModel.SPEC
             else qi.ClassicalModel(mname))
    r = qi.run_one(model, ratio=ratio, seed=seed,
                   X_base=X, y_base=y,
                   n_train=n_train, n_val=n_val, n_test=n_test,
                   verbose=False)
    return {
        'noise': noise,
        'model': r.model, 'ratio': r.ratio, 'seed': r.seed,
        'metrics': r.metrics, 'threshold': r.threshold,
        'fit_time_s': r.fit_time_s, 'n_circuit_evals': r.n_circuit_evals,
        'n_train_majority': r.n_train_majority,
        'n_train_minority': r.n_train_minority,
    }


# =============================================================================
# Main: resume from cache, run the missing tasks, checkpoint periodically
# =============================================================================

def main():
    # -- Load any prior runs ------------------------------------------------
    all_runs = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            partial = json.load(f)
        all_runs.extend(partial.get('runs', []))
        print(f'Resumed: loaded {len(all_runs)} runs from {OUT_PATH}')

    done_keys = {(r['noise'], r['model'], r['ratio'], r['seed']) for r in all_runs}

    # -- Build the task list -----------------------------------------------
    # Order classical/fast models first so the grid produces an interpretable
    # output if interrupted early.
    fast_first = (['data_reuploading'] + CLASSICAL_MODELS
                  + ['fixed_vqc', 'trainable_vqc'])
    tasks = [
        (noise, mname, ratio, seed)
        for noise in NOISE_LEVELS
        for mname in fast_first
        for ratio in RATIOS
        for seed in ALL_SEEDS
        if (noise, mname, ratio, seed) not in done_keys
    ]

    print(f'Already done : {len(done_keys)}')
    print(f'Remaining    : {len(tasks)}')
    print(f'Workers      : N_JOBS={N_JOBS}'
          + ('  [sequential, bit-identical to reference]' if N_JOBS == 1 else ''))

    if not tasks:
        print('All tasks done; nothing to do.')
        return

    def save_checkpoint():
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH + '.tmp', 'w') as f:
            json.dump({'config': CONFIG_OUT, 'runs': all_runs},
                      f, indent=1, default=float)
        os.replace(OUT_PATH + '.tmp', OUT_PATH)

    # -- Run in chunks so we checkpoint every CHUNK_SIZE tasks -------------
    t_start = time.time()
    n_done = 0
    pool = Parallel(n_jobs=N_JOBS, backend='loky', batch_size=1)

    for chunk_start in range(0, len(tasks), CHUNK_SIZE):
        chunk = tasks[chunk_start:chunk_start + CHUNK_SIZE]
        results = pool(delayed(run_task)(*t) for t in chunk)
        all_runs.extend(results)
        save_checkpoint()

        n_done += len(results)
        elapsed = time.time() - t_start
        eta = elapsed / n_done * (len(tasks) - n_done) if n_done > 0 else 0
        last = results[-1]
        print(f'  [{n_done:4d}/{len(tasks)}] noise={last["noise"]} '
              f'{last["model"]:<18s} r={last["ratio"]:.2f} seed={last["seed"]:>2d}  '
              f'BA={last["metrics"]["balanced_accuracy"]:.3f}  '
              f'(elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m)  '
              f'[checkpoint -> {OUT_PATH}]')

    elapsed = time.time() - t_start
    print(f'\nDone. Generated {len(tasks)} new runs in {elapsed/60:.1f} min.')
    print(f'Total runs (cached + new): {len(all_runs)}')
    print(f'Saved to {OUT_PATH}')


if __name__ == '__main__':
    main()
