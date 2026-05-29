"""
smoke_test.py
=============
End-to-end verification that the refactor (qml_core.py + qml_imbalance.py)
works correctly. Run with:

    python3 smoke_test.py

Expected output: every section ends with a line containing "OK".

This script does NOT re-run Notebook 1 or the imbalance grid; it only
verifies that:

  1. Both modules import cleanly without SyntaxWarnings.
  2. The shared circuit-eval counter works across module boundaries
     (this is what allows the AD path in Notebook 1 to share its eval
     count with the PS path).
  3. train_ps_run with store_param_history=True populates the params
     trajectory list (needed by the Bloch slider in Notebook 1).
  4. run_one works end-to-end for one quantum and one classical model.
"""

from __future__ import annotations
import warnings
import sys

# Promote SyntaxWarnings to errors so a stale escape sequence breaks the test.
warnings.filterwarnings("error", category=SyntaxWarning)


def section(title: str) -> None:
    print(f"\n--- {title} ---")


# -----------------------------------------------------------------------------
section("1. Clean imports")
# -----------------------------------------------------------------------------
import qml_core
import qml_imbalance as qi
import numpy as np

print(f"  qml_core: {len([n for n in dir(qml_core) if not n.startswith('_')])} public names")
print(f"  qml_imbalance.QuantumModel.SPEC: {list(qi.QuantumModel.SPEC.keys())}")
print(f"  qml_imbalance.ClassicalModel.SPEC: {list(qi.ClassicalModel.SPEC.keys())}")
print("  OK")

# -----------------------------------------------------------------------------
section("2. Counter sharing across modules (Notebook 1 AD path simulation)")
# -----------------------------------------------------------------------------
qml_core.reset_counter()
qml_core.bump_counter(5)
qml_core.bump_counter(3)
n = qml_core.get_counter()
assert n == 8, f"counter sharing broken: got {n}, expected 8"
# Round-trip via qml_imbalance (re-exports counters)
qi.reset_counter()
qi.bump_counter(7) if hasattr(qi, 'bump_counter') else qml_core.bump_counter(7)
n2 = qi.get_counter()
assert n2 == 7, f"re-exported counters out of sync: {n2}"
print(f"  counter shared correctly across modules: 8 then 7")
print("  OK")

# -----------------------------------------------------------------------------
section("3. train_ps_run with store_param_history=True")
# -----------------------------------------------------------------------------
np.random.seed(0)
X_train = np.random.uniform(-np.pi, np.pi, (50, 2))
y_train = np.sign(np.random.randn(50))
X_val = np.random.uniform(-np.pi, np.pi, (20, 2))
y_val = np.sign(np.random.randn(20))

h = qml_core.train_ps_run(
    np.pi * np.random.rand(6), X_train, y_train, X_val, y_val,
    circuit_type='dr', n_iter=5, lr=0.015,
    select_by='val_acc', store_param_history=True,
)
assert 'params' in h, "store_param_history=True did not record params history"
assert len(h['params']) == 6, f"params history length: {len(h['params'])}, expected 6"
assert 'best_val_acc' in h
assert 'best_val_loss' in h
print(f"  history keys: {sorted(h.keys())}")
print(f"  params trajectory length: {len(h['params'])} (5 iter + 1 final)")
print("  OK")

# -----------------------------------------------------------------------------
section("4. End-to-end run_one (one quantum, one classical)")
# -----------------------------------------------------------------------------
X_base, y_base = qi.make_balanced_dataset(n_total=400, noise=0.1, seed=42)

# Quantum: cheapest model (DR), small iter count
qm = qi.QuantumModel('data_reuploading', n_iter=20, lr=0.015)
res_q = qi.run_one(
    qm, ratio=0.7, seed=0,
    X_base=X_base, y_base=y_base,
    n_train=80, n_val=40, n_test=80,
)
print(f"  Quantum: BA={res_q.metrics['balanced_accuracy']:.3f}  "
      f"evals={res_q.n_circuit_evals:,}")
assert res_q.metrics['balanced_accuracy'] > 0.0
assert res_q.n_circuit_evals > 0

# Classical: logreg
cm = qi.ClassicalModel('logreg')
res_c = qi.run_one(
    cm, ratio=0.7, seed=0,
    X_base=X_base, y_base=y_base,
    n_train=80, n_val=40, n_test=80,
)
print(f"  Classical: BA={res_c.metrics['balanced_accuracy']:.3f}  "
      f"evals={res_c.n_circuit_evals}")
assert res_c.metrics['balanced_accuracy'] > 0.5  # logreg should clearly beat trivial
assert res_c.n_circuit_evals == 0  # classical models don't run circuits

# Sanity: metric panel is BA, the threshold-independent ROC AUC byproduct,
# and the four confusion-matrix counts (plus the stored threshold).
expected_keys = {'balanced_accuracy', 'roc_auc', 'tn', 'fp', 'fn', 'tp', 'threshold'}
got_keys = set(res_c.metrics.keys())
assert got_keys == expected_keys, (
    f"compute_metrics keys changed: expected {expected_keys}, got {got_keys}"
)
print(f"  Metric panel OK: {sorted(got_keys)}")
print("  OK")

# -----------------------------------------------------------------------------
section("5. apply_class_weight basic correctness")
# -----------------------------------------------------------------------------
import numpy as np
X = np.zeros((100, 2))
y = np.concatenate([np.zeros(90, dtype=int), np.ones(10, dtype=int)])
X_out, y_out, sw = qi.apply_class_weight(X, y)
# Average weight should be 1
assert np.isclose(sw.mean(), 1.0), f"mean weight = {sw.mean()}, expected 1.0"
# Minority weight should be 9x majority weight (90/10 imbalance, 2 classes)
maj_w = sw[y == 0][0]
min_w = sw[y == 1][0]
ratio_obs = min_w / maj_w
assert np.isclose(ratio_obs, 9.0), f"weight ratio = {ratio_obs}, expected 9.0"
# Data passed through unchanged
assert np.array_equal(X, X_out)
assert np.array_equal(y, y_out)
print(f"  weights: maj={maj_w:.3f}, min={min_w:.3f}, ratio={ratio_obs:.2f} (expected 9.0)")
print("  OK")

# -----------------------------------------------------------------------------
section("6. friedman_wilcoxon_holm minimal call (synthetic)")
# -----------------------------------------------------------------------------
# Build a few synthetic RunResults with known structure
fake = []
np.random.seed(0)
for m in ['a', 'b', 'c']:
    base = {'a': 0.9, 'b': 0.7, 'c': 0.5}[m]
    for s in range(12):
        fake.append(qi.RunResult(
            model=m, ratio=0.5, seed=s,
            metrics={'balanced_accuracy': base + 0.01 * np.random.randn(),
                     'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0, 'threshold': 0.0},
            threshold=0.0, fit_time_s=0.0, n_circuit_evals=0,
            n_train_majority=0, n_train_minority=0,
        ))
out = qi.friedman_wilcoxon_holm(fake, ratio=0.5)
assert out['friedman_p'] < 0.001, f"Friedman p={out['friedman_p']} should be tiny"
sig_pairs = sum(1 for v in out['pairwise'].values() if v['reject_holm'])
print(f"  Friedman p = {out['friedman_p']:.3e}, sig pairs = {sig_pairs}/3")
assert sig_pairs == 3, f"expected all 3 pairs significant, got {sig_pairs}"
print("  OK")

# -----------------------------------------------------------------------------
section("7. Notebook 1 syntax (every code cell parses)")
# -----------------------------------------------------------------------------
import json, ast, os
NB = 'VQC_Autodiff_and_ParameterShift.ipynb'
if os.path.exists(NB):
    with open(NB) as f:
        nb = json.load(f)
    ok = bad = 0
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        try:
            ast.parse(src)
            ok += 1
        except SyntaxError as e:
            bad += 1
            print(f"  Cell {i}: SyntaxError -> {e}")
    print(f"  {ok} code cells parsed OK, {bad} with syntax errors")
    assert bad == 0, "Notebook has syntax errors"
    print("  OK")
else:
    print(f"  (skipped: {NB} not found in cwd)")

print("\n[ALL CHECKS PASSED]")
sys.exit(0)
