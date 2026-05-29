"""
build_notebooks.py
==================
Generator for the four annex notebooks (one per toy dataset). Instead of
re-implementing the Case Study 2 notebook, each annex notebook is produced by
**transforming** the original `code/imbalance_experiments.ipynb`: every analysis
cell is copied verbatim and only the data source and the output paths are
patched. This guarantees the experiment and every figure/table/statistic are
byte-identical to the main study, so the annex numbers are directly comparable;
the only thing that changes is the base dataset (a toy dataset from
`toy_datasets` instead of `make_moons`).

Patches applied (and nothing else):
  * imports cell: add the parent dir to sys.path, import `toy_datasets`, and
    define the `DATASET` selector;
  * dataset-generation and ablation cells: `qi.make_balanced_dataset(...)` ->
    `td.make_dataset(DATASET, ...)`;
  * `RESULTS_PATH` -> the per-dataset cache `results/annex_<dataset>.json`
    (produced by `run_annex.py`, same JSON format as the main study);
  * figure paths `figures/case2-*` -> `figures/<dataset>-case2-*` (so the four
    notebooks do not clobber each other);
  * the two interpretive "Reading" markdown cells of the ablation section are
    replaced by a one-line note, because their prose quotes make_moons-specific
    numbers; the ablation *code* still runs on the new dataset;
  * a banner cell at the top, and `make_moons` -> the dataset name in the prose.

Re-run after editing the original notebook or this script:
    python build_notebooks.py
"""

from __future__ import annotations

import copy
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, _CODE)
from toy_datasets import DATASETS, DATASET_INFO  # noqa: E402

ORIGINAL = os.path.join(_CODE, 'imbalance_experiments.ipynb')

# Exact substrings to patch in CODE cells (data source + paths only).
MAKE_BALANCED_SRC = ("qi.make_balanced_dataset(\n"
                     "        n_total=CONFIG['n_total_base'], noise=noise, seed=42)")
MAKE_BALANCED_DST = ("td.make_dataset(\n"
                     "        DATASET, n_samples=CONFIG['n_total_base'], "
                     "noise=noise, random_state=42)")

# Markers identifying the two ablation "Reading" markdown cells to neutralise.
ABLATION_READING_MARKERS = (
    "threshold tuning alone",     # SVM ablation reading
    "trainable VQC inverts",      # VQC ablation reading
)
ABLATION_NOTE = (
    "**Reading.** The table above is the same ablation as the main study, "
    "recomputed on the **{title}** dataset, so the *mechanism* (how much of the "
    "headline BA at $r=0.99$ comes from `class_weight` vs. threshold tuning) is "
    "directly comparable. The numeric commentary for `make_moons` lives in "
    "`../imbalance_experiments.ipynb`; here, read the two columns off the table "
    "for {title}.")


def banner(ds: str) -> dict:
    title = ds.capitalize()
    return {"cell_type": "markdown", "id": "annex-banner", "metadata": {},
            "source": (
f"""> **Dataset-robustness annex - {title} dataset.** This notebook is an
> *exact* copy of `../imbalance_experiments.ipynb` (Case Study 2): same grid
> (2 noise levels x 8 models x 7 ratios x 12 seeds), same models, same splits,
> same hyperparameters, same analysis code and same statistical protocol
> (Friedman + Wilcoxon + Holm). **Only the base dataset changes**: it is drawn
> from `toy_datasets.make_{ds}` ({DATASET_INFO[ds]}) instead of `make_moons`,
> so every number here is directly comparable to the main study. Results load
> from `results/annex_{ds}.json` (produce it with `python run_annex.py {ds}`).
""").splitlines(keepends=True)}


def patch_code(src: str, ds: str) -> str:
    title = ds.capitalize()
    # 1. data source (cells 6, 30, 32)
    src = src.replace(MAKE_BALANCED_SRC, MAKE_BALANCED_DST)
    # 2. results path (cell 10)
    src = src.replace("RESULTS_PATH = 'results/runs_simplified.json'",
                      "RESULTS_PATH = f'results/annex_{DATASET}.json'")
    # 3. figure paths (both plain and f-string forms)
    src = src.replace("plt.savefig(f'figures/case2-",
                      "plt.savefig(f'figures/{DATASET}-case2-")
    src = src.replace("plt.savefig('figures/case2-",
                      "plt.savefig(f'figures/{DATASET}-case2-")
    # 4. dataset title in the base-visualisation cell
    src = src.replace("f'make_moons(noise={noise})  n={len(X)}'",
                      "f'{DATASET}(noise={noise})  n={len(X)}'")
    src = src.replace("'The two datasets used in this case study'",
                      "f'The {DATASET} base dataset at both noise levels'")
    # 5. imports cell: parent path + toy_datasets + DATASET selector
    if "import qml_imbalance as qi" in src and "sys.path.insert" in src:
        src = src.replace(
            "sys.path.insert(0, os.path.abspath('.'))",
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath('.'))")
        src = src.replace(
            "import qml_imbalance as qi\n",
            "import qml_imbalance as qi\n"
            "import toy_datasets as td\n\n"
            f"DATASET = {ds!r}\n")
    return src


def transform() -> dict:
    with open(ORIGINAL, encoding='utf-8') as f:
        original = json.load(f)
    return original


def build_notebook(ds: str, original: dict) -> dict:
    title = ds.capitalize()
    nb = copy.deepcopy(original)
    new_cells = [banner(ds)]
    cid = 0
    for cell in nb['cells']:
        cid += 1
        cell = copy.deepcopy(cell)
        cell.setdefault('id', f'orig{cid:03d}')
        src = ''.join(cell['source'])
        if cell['cell_type'] == 'code':
            src = patch_code(src, ds)
        else:  # markdown
            if any(m in src for m in ABLATION_READING_MARKERS):
                src = ABLATION_NOTE.format(title=title)
            else:
                src = src.replace('make_moons', ds)
        cell['source'] = src.splitlines(keepends=True)
        # never ship stale outputs/exec counts in the template
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        new_cells.append(cell)
    nb['cells'] = new_cells
    nb.setdefault('metadata', {})
    nb['metadata'].setdefault('kernelspec',
                              {"display_name": "Python 3", "language": "python",
                               "name": "python3"})
    return nb


def main():
    original = transform()
    for ds in DATASETS:
        nb = build_notebook(ds, original)
        path = os.path.join(_HERE, f'annex_{ds}.ipynb')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=True)
        print(f'wrote {path}  ({len(nb["cells"])} cells)')


if __name__ == '__main__':
    main()
