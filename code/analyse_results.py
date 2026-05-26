"""analyse_results.py
--------------------
Run the pre-registered analysis pipeline on results/runs_simplified.json
and dump every number needed by Case Study 2 of the thesis to stdout.
"""
import os, sys, json
import numpy as np
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qml_imbalance as qi

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, 'results', 'runs_simplified.json')) as f:
    data = json.load(f)
CONFIG = data['config']
runs = data['runs']

NOISE_LEVELS = CONFIG['noise_levels']
RATIOS = CONFIG['ratios']
ALL_MODELS = CONFIG['quantum_models'] + CONFIG['classical_models']
EXTREME = max(RATIOS)
BALANCED = min(RATIOS)

results_by_noise = defaultdict(list)
for r in runs:
    rr = qi.RunResult(
        model=r['model'], ratio=r['ratio'], seed=r['seed'],
        metrics=r['metrics'], threshold=r['threshold'],
        fit_time_s=r['fit_time_s'],
        n_circuit_evals=r['n_circuit_evals'],
        n_train_majority=r['n_train_majority'],
        n_train_minority=r['n_train_minority'],
    )
    results_by_noise[r['noise']].append(rr)

# ============================================================================
# 1. Aggregated balanced accuracy tables
# ============================================================================
print('=' * 90)
print('1. Mean BA across 12 seeds, per (noise, model, ratio)')
print('=' * 90)
for noise in NOISE_LEVELS:
    print(f'\n  noise = {noise}')
    agg = qi.aggregate(results_by_noise[noise], by=('model', 'ratio'))
    header = f'  {"model":<20s}' + ''.join(f'r={r:.2f}'.rjust(13)
                                             for r in RATIOS)
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for m in ALL_MODELS:
        line = f'  {m:<20s}'
        for r in RATIOS:
            row = next((x for x in agg
                        if x['model'] == m and x['ratio'] == r), None)
            if row:
                line += (f'{row["balanced_accuracy_mean"]:.3f}±'
                          f'{row["balanced_accuracy_std"]:.2f}').rjust(13)
            else:
                line += 'na'.rjust(13)
        print(line)

# ============================================================================
# 2. Robustness slopes
# ============================================================================
print('\n' + '=' * 90)
print(f'2. Robustness slope  Δ = BA(r={BALANCED}) - BA(r={EXTREME}),  '
      'sorted ascending')
print('=' * 90)
for noise in NOISE_LEVELS:
    print(f'\n  noise = {noise}')
    print(f'  {"rank":<5}{"model":<22}{"BA(0.5)":>10}{"BA(0.99)":>10}'
          f'{"Δ":>10}')
    print('  ' + '-' * 52)
    rows = []
    for m in ALL_MODELS:
        vs_lo = [r.metrics['balanced_accuracy']
                  for r in results_by_noise[noise]
                  if r.model == m and r.ratio == BALANCED]
        vs_hi = [r.metrics['balanced_accuracy']
                  for r in results_by_noise[noise]
                  if r.model == m and r.ratio == EXTREME]
        if not vs_lo or not vs_hi:
            continue
        ba_lo = float(np.mean(vs_lo))
        ba_hi = float(np.mean(vs_hi))
        rows.append((m, ba_lo, ba_hi, ba_lo - ba_hi))
    rows.sort(key=lambda t: t[-1])
    for rank, (m, lo, hi, d) in enumerate(rows, 1):
        print(f'  {rank:<5}{m:<22}{lo:>10.3f}{hi:>10.3f}{d:>+10.3f}')

# ============================================================================
# 3. Friedman omnibus per ratio
# ============================================================================
print('\n' + '=' * 90)
print('3. Friedman omnibus chi^2 and p-value per (noise, ratio)')
print('=' * 90)
print(f'  {"noise":<7}{"r":<6}{"chi^2":>12}{"p":>14}{"reject @ 0.05":>16}')
print('  ' + '-' * 55)
fwh_tables = {}
for noise in NOISE_LEVELS:
    fwh_tables[noise] = qi.friedman_wilcoxon_holm(results_by_noise[noise])
    for r in sorted(fwh_tables[noise].keys()):
        st = fwh_tables[noise][r]
        rej = 'YES' if st['friedman_p'] < 0.05 else 'no'
        print(f'  {noise:<7}{r:<6.2f}{st["friedman_stat"]:>12.3f}'
              f'{st["friedman_p"]:>14.3e}{rej:>16}')

# ============================================================================
# 4. Holm-significant pair count per (noise, ratio)
# ============================================================================
print('\n' + '=' * 90)
print('4. Number of Holm-corrected significant pairs per (noise, ratio)')
print(f'   Floor at n=12: smallest Holm-p = 28 * 2/2^12 = {28*2/2**12:.4f}')
print('=' * 90)
print(f'  {"noise":<7}{"r":<6}{"# sig pairs (of 28)":>22}')
for noise in NOISE_LEVELS:
    for r in sorted(fwh_tables[noise].keys()):
        st = fwh_tables[noise][r]
        n_sig = sum(1 for v in st['pairwise'].values() if v['reject_holm'])
        print(f'  {noise:<7}{r:<6.2f}{n_sig:>15} / 28')

# ============================================================================
# 5. Quantum-advantage matrices per noise level
# ============================================================================
print('\n' + '=' * 90)
print('5. Quantum-advantage matrices: count of ratios (of 7) where '
      'Wilcoxon+Holm certifies Q > C')
print('=' * 90)

def q_beats_c(stats_per_ratio, qm, cm, r, alpha=0.05):
    st = stats_per_ratio[r]
    if np.isnan(st['friedman_p']) or st['friedman_p'] >= alpha:
        return False
    pair = st['pairwise']
    if (qm, cm) in pair:
        info = pair[(qm, cm)]
        return info['reject_holm'] and info['mean_diff'] > 0
    if (cm, qm) in pair:
        info = pair[(cm, qm)]
        return info['reject_holm'] and info['mean_diff'] < 0
    return False

for noise in NOISE_LEVELS:
    print(f'\n  noise = {noise}')
    COLW = 14
    print(f'  {"Q model":<20}' + ''.join(c.rjust(COLW)
                                           for c in CONFIG['classical_models']))
    print('  ' + '-' * (20 + COLW * len(CONFIG['classical_models'])))
    for qm in CONFIG['quantum_models']:
        line = f'  {qm:<20}'
        for cm in CONFIG['classical_models']:
            wins = sum(1 for r in RATIOS
                        if q_beats_c(fwh_tables[noise], qm, cm, r))
            line += f'{wins}/7'.rjust(COLW)
        print(line)

# ============================================================================
# 6. Reverse: count of ratios where C > Q significantly
# ============================================================================
print('\n' + '=' * 90)
print('6. Reverse matrices: count of ratios where Wilcoxon+Holm certifies C > Q')
print('=' * 90)
for noise in NOISE_LEVELS:
    print(f'\n  noise = {noise}')
    COLW = 14
    print(f'  {"Q model":<20}' + ''.join(c.rjust(COLW)
                                           for c in CONFIG['classical_models']))
    print('  ' + '-' * (20 + COLW * len(CONFIG['classical_models'])))
    for qm in CONFIG['quantum_models']:
        line = f'  {qm:<20}'
        for cm in CONFIG['classical_models']:
            wins = 0
            for r in RATIOS:
                st = fwh_tables[noise][r]
                if np.isnan(st['friedman_p']) or st['friedman_p'] >= 0.05:
                    continue
                pair = st['pairwise']
                if (qm, cm) in pair:
                    info = pair[(qm, cm)]
                    if info['reject_holm'] and info['mean_diff'] < 0:
                        wins += 1
                elif (cm, qm) in pair:
                    info = pair[(cm, qm)]
                    if info['reject_holm'] and info['mean_diff'] > 0:
                        wins += 1
            line += f'{wins}/7'.rjust(COLW)
        print(line)

# ============================================================================
# 7. Pairwise table at r=0.99 (the headline)
# ============================================================================
for noise in NOISE_LEVELS:
    print('\n' + '=' * 90)
    print(f'7. Pairwise Wilcoxon+Holm at noise={noise}, r={EXTREME}')
    print('=' * 90)
    qi.print_friedman_summary(fwh_tables[noise][EXTREME], alpha=0.05)

# ============================================================================
# 8. Cost (fit time + circuit evals) per (noise, model)
# ============================================================================
print('\n' + '=' * 90)
print('8. Compute cost')
print('=' * 90)
for noise in NOISE_LEVELS:
    print(f'\n  noise = {noise}')
    print(f'  {"model":<22}{"fit_mean (s)":>14}{"fit_med (s)":>14}'
          f'{"circuit_evals_mean":>22}')
    print('  ' + '-' * 72)
    rows = []
    for m in ALL_MODELS:
        ts = [r.fit_time_s for r in results_by_noise[noise] if r.model == m]
        ev = [r.n_circuit_evals for r in results_by_noise[noise] if r.model == m]
        if not ts:
            continue
        rows.append((m, float(np.mean(ts)), float(np.median(ts)),
                       int(np.mean(ev)) if ev else 0))
    rows.sort(key=lambda t: -t[1])
    for m, mu, md_, ev in rows:
        print(f'  {m:<22}{mu:>14.2f}{md_:>14.2f}{ev:>22,}')

# ============================================================================
# 9. Confusion-matrix breakdown at r=0.99, seed=0
# ============================================================================
print('\n' + '=' * 90)
print('9. Confusion-matrix breakdown at r=0.99, seed=0')
print('=' * 90)
for noise in NOISE_LEVELS:
    print(f'\n  noise = {noise}')
    print(f'  {"model":<22}{"BA":>8}{"TN":>6}{"FP":>6}{"FN":>6}{"TP":>6}'
          f'{"FPR":>8}{"FNR":>8}')
    print('  ' + '-' * 70)
    for m in ALL_MODELS:
        m_runs = [r for r in results_by_noise[noise]
                  if r.model == m and r.ratio == EXTREME and r.seed == 0]
        if not m_runs:
            continue
        rr = m_runs[0]
        tn, fp, fn, tp = (rr.metrics['tn'], rr.metrics['fp'],
                          rr.metrics['fn'], rr.metrics['tp'])
        fpr = fp / (fp + tn) if (fp + tn) else 0
        fnr = fn / (fn + tp) if (fn + tp) else 0
        print(f'  {m:<22}{rr.metrics["balanced_accuracy"]:>8.3f}'
              f'{tn:>6}{fp:>6}{fn:>6}{tp:>6}{fpr:>8.2%}{fnr:>8.2%}')
