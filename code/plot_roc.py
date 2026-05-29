"""plot_roc.py
---------------
Threshold-free visual companion to analyse_auc.py. Re-runs the Case Study 2
grid for the two extreme imbalance ratios (balanced r=0.5 and severe r=0.99)
at both noise levels, captures the per-sample test scores, and plots the
mean ROC curve over the 12 seeds per (noise, ratio, model) cell.

Why ROC and not just AUC?
    AUC collapses the whole ranking quality into one scalar. The shape of
    the curve is what tells you *where* on the operating range the model
    pays for the imbalance: a curve that hugs the y-axis is good at
    cheap recall (low FPR), a curve that bulges only near (1,1) is good
    only at expensive recall. Two models with identical AUC can have
    very different shapes and therefore very different practical value.

The runs reproduce exactly the ones in results/runs_simplified.json (same
seeds, same handler, same hyperparameters), so the AUC values on the curves
match the table in analyse_auc.py to within float noise.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qml_imbalance as qi

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Match the production grid used by runs_simplified.json.
NOISE_LEVELS = [0.1, 0.3]
RATIOS = [0.5, 0.99]
SEEDS = list(range(12))
QUANTUM = ['data_reuploading', 'fixed_vqc', 'trainable_vqc']
CLASSICAL = ['logreg', 'svm_rbf', 'knn', 'rf', 'mlp_small']
ALL_MODELS = QUANTUM + CLASSICAL
N_TRAIN, N_VAL, N_TEST = 500, 400, 1200
N_TOTAL_BASE = 8000

# Common FPR grid for averaging curves across seeds.
FPR_GRID = np.linspace(0.0, 1.0, 201)

# Colour palette: distinct hues for quantum vs classical, plus per-model variation.
COLOURS = {
    'data_reuploading': '#d62728',
    'fixed_vqc':        '#ff7f0e',
    'trainable_vqc':    '#bcbd22',
    'logreg':           '#1f77b4',
    'svm_rbf':          '#2ca02c',
    'knn':              '#9467bd',
    'rf':               '#8c564b',
    'mlp_small':        '#17becf',
}
LINESTYLE = {m: '-' for m in QUANTUM}
LINESTYLE.update({m: '--' for m in CLASSICAL})


def make_model(name: str):
    if name in QUANTUM:
        return qi.QuantumModel(name, n_iter=200, lr=0.015)
    return qi.ClassicalModel(name)


def run_and_score(model_name: str, noise: float, ratio: float, seed: int):
    """Re-run one (model, noise, ratio, seed) experiment and return
    (y_test, scores_test). Exactly mirrors qi.run_one but exposes the
    continuous test scores instead of only the aggregated metrics."""
    X_base, y_base = qi.make_balanced_dataset(
        n_total=N_TOTAL_BASE, noise=noise, seed=42)
    X_tr, y_tr, X_va, y_va, X_te, y_te = qi.make_imbalanced_split(
        X_base, y_base, train_ratio_majority=ratio,
        n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=seed,
    )
    X_tr_h, y_tr_h, sw_tr_h = qi.apply_class_weight(X_tr, y_tr)
    model = make_model(model_name)
    if model.kind == 'quantum':
        X_tr_use, X_va_use, X_te_use = qi.rescale_to_pi(X_tr_h, X_va, X_te)
    else:
        X_tr_use, X_va_use, X_te_use = X_tr_h, X_va, X_te
    state = model.fit(X_tr_use, y_tr_h, sw_tr_h, X_va_use, y_va, seed=seed)
    scores_test = model.predict_score(state, X_te_use)
    return y_te, np.asarray(scores_test)


def mean_roc(curves):
    """Vertical-averaging of ROC curves over seeds: interpolate TPR at a
    common FPR grid (the standard scikit-learn recipe). Returns mean TPR,
    std TPR, and mean AUC across seeds."""
    tprs = []
    aucs = []
    for fpr, tpr in curves:
        t = np.interp(FPR_GRID, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
        aucs.append(sk_auc(fpr, tpr))
    tprs = np.vstack(tprs)
    return tprs.mean(axis=0), tprs.std(axis=0), float(np.mean(aucs))


def main():
    fig, axes = plt.subplots(
        len(NOISE_LEVELS), len(RATIOS),
        figsize=(11, 9), sharex=True, sharey=True,
    )

    for i, noise in enumerate(NOISE_LEVELS):
        for j, ratio in enumerate(RATIOS):
            ax = axes[i, j]
            print(f'[noise={noise}  ratio={ratio}]')
            for m in ALL_MODELS:
                curves = []
                for s in SEEDS:
                    y_te, scores = run_and_score(m, noise, ratio, s)
                    fpr, tpr, _ = roc_curve(y_te, scores)
                    curves.append((fpr, tpr))
                tpr_m, tpr_s, auc_m = mean_roc(curves)
                ax.plot(FPR_GRID, tpr_m,
                        color=COLOURS[m], linestyle=LINESTYLE[m],
                        linewidth=1.6,
                        label=f'{m} (AUC={auc_m:.3f})')
                ax.fill_between(FPR_GRID, tpr_m - tpr_s, tpr_m + tpr_s,
                                color=COLOURS[m], alpha=0.08, linewidth=0)
                print(f'  {m:<20s} AUC={auc_m:.3f}')
            ax.plot([0, 1], [0, 1], color='gray', linewidth=0.8,
                    linestyle=':')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.01)
            ax.set_title(f'noise={noise},  train majority ratio={ratio}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=7, framealpha=0.9)

    for ax in axes[-1, :]:
        ax.set_xlabel('False positive rate')
    for ax in axes[:, 0]:
        ax.set_ylabel('True positive rate')

    fig.suptitle('Test-set ROC curves, mean over 12 seeds (band: ±1 std)',
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGDIR, 'case2-roc-curves.png')
    fig.savefig(out, dpi=150)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
