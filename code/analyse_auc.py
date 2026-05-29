"""analyse_auc.py
----------------
Threshold-independent robustness check for Case Study 2. Runs the same
descriptive + pre-registered statistical machinery as analyse_results.py but
on the ROC AUC byproduct instead of balanced accuracy, and contrasts the two.

ROC AUC is NOT a pre-registered headline metric: it is reported in the thesis
only as a control on the BA-based verdict, because the ablation shows that
threshold tuning on the balanced validation set drives much of the headline BA
at extreme imbalance. AUC removes thresholding from the picture entirely.
"""
import os, sys, json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qml_imbalance as qi

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, 'results', 'runs_simplified.json')) as f:
    data = json.load(f)
CONFIG = data['config']
runs = data['runs']

NOISE_LEVELS = CONFIG['noise_levels']
RATIOS = CONFIG['ratios']
QUANTUM = CONFIG['quantum_models']
CLASSICAL = CONFIG['classical_models']
ALL_MODELS = QUANTUM + CLASSICAL
EXTREME = max(RATIOS)
BALANCED = min(RATIOS)

by_noise = defaultdict(list)
for r in runs:
    rr = qi.RunResult(
        model=r['model'], ratio=r['ratio'], seed=r['seed'],
        metrics=r['metrics'], threshold=r['threshold'],
        fit_time_s=r['fit_time_s'], n_circuit_evals=r['n_circuit_evals'],
        n_train_majority=r['n_train_majority'],
        n_train_minority=r['n_train_minority'])
    by_noise[r['noise']].append(rr)


def mean_metric(noise, model, ratio, metric):
    vs = [r.metrics[metric] for r in by_noise[noise]
          if r.model == model and r.ratio == ratio]
    return float(np.mean(vs)) if vs else float('nan')


def std_metric(noise, model, ratio, metric):
    vs = [r.metrics[metric] for r in by_noise[noise]
          if r.model == model and r.ratio == ratio]
    return float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0


# 1. AUC table
print('=' * 100)
print('1. Mean ROC AUC (mean +/- std over 12 seeds) per (noise, model, ratio)')
print('=' * 100)
for noise in NOISE_LEVELS:
    print(f'\n  noise = {noise}')
    header = f'  {"model":<20s}' + ''.join(f'r={r:.2f}'.rjust(13) for r in RATIOS)
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for m in ALL_MODELS:
        line = f'  {m:<20s}'
        for r in RATIOS:
            line += (f'{mean_metric(noise,m,r,"roc_auc"):.3f}'
                     f'±{std_metric(noise,m,r,"roc_auc"):.2f}').rjust(13)
        print(line)

# 2. AUC robustness slope
print('\n' + '=' * 100)
print(f'2. AUC robustness slope  D = AUC(r={BALANCED}) - AUC(r={EXTREME}), ascending')
print('=' * 100)
for noise in NOISE_LEVELS:
    print(f'\n  noise = {noise}')
    print(f'  {"rank":<5}{"model":<22}{"AUC(0.5)":>10}{"AUC(0.99)":>11}{"D":>10}')
    rows = []
    for m in ALL_MODELS:
        lo = mean_metric(noise, m, BALANCED, 'roc_auc')
        hi = mean_metric(noise, m, EXTREME, 'roc_auc')
        rows.append((m, lo, hi, lo - hi))
    rows.sort(key=lambda t: t[-1])
    for rank, (m, lo, hi, d) in enumerate(rows, 1):
        print(f'  {rank:<5}{m:<22}{lo:>10.3f}{hi:>11.3f}{d:>+10.3f}')

# 3. AUC Friedman + Holm-significant counts
print('\n' + '=' * 100)
print('3. AUC Friedman omnibus + #Holm-significant pairs per (noise, ratio)')
print('=' * 100)
fwh = {}
for noise in NOISE_LEVELS:
    fwh[noise] = qi.friedman_wilcoxon_holm(by_noise[noise], metric='roc_auc')
    print(f'\n  noise = {noise}')
    print(f'  {"r":<6}{"chi^2":>12}{"p":>14}{"reject":>9}{"#sig/28":>10}')
    for r in sorted(fwh[noise].keys()):
        st = fwh[noise][r]
        nsig = sum(1 for v in st['pairwise'].values() if v['reject_holm'])
        rej = 'YES' if st['friedman_p'] < 0.05 else 'no'
        print(f'  {r:<6.2f}{st["friedman_stat"]:>12.3f}{st["friedman_p"]:>14.3e}'
              f'{rej:>9}{nsig:>7}/28')

# 4. AUC quantum-advantage matrices (Q>C and C>Q)
def q_beats_c(stats_per_ratio, qm, cm, r, sign, alpha=0.05):
    st = stats_per_ratio[r]
    if np.isnan(st['friedman_p']) or st['friedman_p'] >= alpha:
        return False
    pair = st['pairwise']
    if (qm, cm) in pair:
        info = pair[(qm, cm)]
        return info['reject_holm'] and (info['mean_diff'] > 0) == (sign > 0)
    if (cm, qm) in pair:
        info = pair[(cm, qm)]
        return info['reject_holm'] and (info['mean_diff'] < 0) == (sign > 0)
    return False

for sign, title in [(+1, 'Q > C'), (-1, 'C > Q')]:
    print('\n' + '=' * 100)
    print(f'4. AUC advantage matrix: #ratios (of 7) where Wilcoxon+Holm certifies {title}')
    print('=' * 100)
    for noise in NOISE_LEVELS:
        print(f'\n  noise = {noise}')
        print(f'  {"Q model":<20}' + ''.join(c.rjust(14) for c in CLASSICAL))
        for qm in QUANTUM:
            line = f'  {qm:<20}'
            for cm in CLASSICAL:
                wins = sum(1 for r in RATIOS
                           if q_beats_c(fwh[noise], qm, cm, r, sign))
                line += f'{wins}/7'.rjust(14)
            print(line)

# 5. BA vs AUC side by side for the quantum models at r=0.5 and r=0.99
print('\n' + '=' * 100)
print('5. BA vs AUC contrast for all models (r=0.50 and r=0.99)')
print('=' * 100)
for noise in NOISE_LEVELS:
    print(f'\n  noise = {noise}')
    print(f'  {"model":<20}{"BA@.50":>9}{"AUC@.50":>9}{"BA@.99":>9}{"AUC@.99":>9}')
    for m in ALL_MODELS:
        print(f'  {m:<20}'
              f'{mean_metric(noise,m,0.5,"balanced_accuracy"):>9.3f}'
              f'{mean_metric(noise,m,0.5,"roc_auc"):>9.3f}'
              f'{mean_metric(noise,m,0.99,"balanced_accuracy"):>9.3f}'
              f'{mean_metric(noise,m,0.99,"roc_auc"):>9.3f}')
