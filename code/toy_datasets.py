"""
toy_datasets.py
===============
Synthetic 2-D binary-classification datasets used as the *robustness annex*
of Case Study 2 of the thesis. They reproduce the four toy datasets defined in
Appendix E.1 of Rodriguez-Grasa, Ban & Sanz, "Neural quantum kernels: training
quantum kernels with quantum neural networks" (arXiv:2401.04642v2):

    sinus    -- points above / below the curve f(x1) = -0.8 sin(pi x1)
    corners  -- four quarter-circles of radius 0.75 at the corners of [-1, 1]^2
    spiral   -- two interleaved Archimedean spirals
    circles  -- an annular region between two concentric circles

Why this module exists
----------------------
The Case Study 2 pipeline (`qml_imbalance.run_one`, `make_imbalanced_split`,
`rescale_to_pi`, ...) is fully parametrised by a *balanced* base dataset of the
form ``(X, y)`` with ``X`` of shape ``(N, 2)`` and ``y`` in ``{0, 1}``. The main
study draws that base from scikit-learn's ``make_moons``; this module provides
four drop-in alternatives so the same imbalance experiment can be re-run on
geometrically different problems and we can check whether the quantum-vs-classical
verdict transfers (thesis Annex).

Interface
---------
Every generator mirrors the scikit-learn convention so it can be swapped for
``make_moons`` with no other change:

    X, y = make_sinus(n_samples=2000, noise=0.1, random_state=None)

    X : ndarray (n_samples, 2)   features, roughly in [-1, 1]^2
    y : ndarray (n_samples,)     labels in {0, 1}, exactly 50/50 balanced

`noise` is the standard deviation of i.i.d. Gaussian jitter added to the point
coordinates *after* labelling, exactly as ``make_moons`` does it: it controls
how much the two classes overlap near their decision boundary. The default 0.1
matches the low-overlap noise level of the main study.

Label convention. The paper labels the two classes ``+1`` / ``-1``; here we use
``{0, 1}`` to match ``make_moons`` and the rest of the framework. The mapping is
``+1 -> 1`` and ``-1 -> 0``. Because the datasets are forced to be exactly
balanced, which class is "0" and which is "1" is immaterial to the experiment
(``make_imbalanced_split`` treats class 0 as the majority and class 1 as the
minority regardless of geometry).

Balance. Unlike the raw region definitions in the paper -- whose class
proportions follow the areas of the regions and are only approximately balanced
-- every generator here returns an *exactly* 50/50 dataset (via per-class
rejection sampling for the region-defined datasets, and by construction for the
spiral). This keeps the "balanced base, then artificially imbalanced" design of
Case Study 2 intact: the imbalance is imposed by the split, not inherited from
the dataset geometry.

A `DATASETS` registry and a `make_dataset(name, ...)` dispatcher are provided so
notebooks and runners can iterate over the four problems uniformly.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

__all__ = [
    "make_sinus",
    "make_corners",
    "make_spiral",
    "make_circles",
    "make_dataset",
    "DATASETS",
    "DATASET_INFO",
]


# =============================================================================
# Internal helper: balanced rejection sampler for region-defined datasets
# =============================================================================

def _balanced_by_rejection(
    label_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_samples: int,
    noise: float,
    rng: np.random.RandomState,
    bound: float = 1.0,
    max_batches: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw an exactly 50/50 dataset whose labels come from ``label_fn``.

    Points are sampled uniformly on the square ``[-bound, bound]^2`` and labelled
    by ``label_fn(x1, x2) -> {0, 1}`` (vectorised over the batch). Two buckets are
    filled independently until each holds ``n_samples // 2`` points, so the class
    balance is exact regardless of how the regions partition the square. Gaussian
    jitter of standard deviation ``noise`` is added *after* labelling, so points
    near the boundary may cross it -- this is the same kind of label overlap that
    ``make_moons(noise=...)`` produces.

    If ``n_samples`` is odd, class 0 gets the extra point.
    """
    n1 = n_samples // 2
    n0 = n_samples - n1
    buckets: Dict[int, list] = {0: [], 1: []}
    targets = {0: n0, 1: n1}
    batch = max(2048, 4 * n_samples)

    for _ in range(max_batches):
        if len(buckets[0]) >= n0 and len(buckets[1]) >= n1:
            break
        pts = rng.uniform(-bound, bound, size=(batch, 2))
        lab = label_fn(pts[:, 0], pts[:, 1]).astype(int)
        for c in (0, 1):
            need = targets[c] - len(buckets[c])
            if need > 0:
                sel = pts[lab == c]
                if len(sel):
                    buckets[c].extend(sel[:need])
    else:
        raise RuntimeError(
            "rejection sampler did not converge; one class may be too rare "
            "for the requested region definition"
        )

    X = np.vstack([np.array(buckets[0]), np.array(buckets[1])])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    if noise:
        X = X + rng.normal(0.0, noise, size=X.shape)

    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def _as_rng(random_state) -> np.random.RandomState:
    if isinstance(random_state, np.random.RandomState):
        return random_state
    return np.random.RandomState(random_state)


# =============================================================================
# 1. Sinus
# =============================================================================

def make_sinus(n_samples: int = 2000, noise: float = 0.1,
               random_state=None) -> Tuple[np.ndarray, np.ndarray]:
    """Sinus dataset (Appendix E.1).

    The decision boundary is the curve ``f(x1) = -0.8 sin(pi x1)`` on the square
    ``[-1, 1]^2``. Points *below* the curve are class 1 (paper's +1); points
    *above* it are class 0 (paper's -1). Returns an exactly balanced ``(X, y)``.
    """
    rng = _as_rng(random_state)

    def label(x1, x2):
        f = -0.8 * np.sin(np.pi * x1)
        return (x2 < f).astype(int)   # below curve -> +1 -> 1

    return _balanced_by_rejection(label, n_samples, noise, rng, bound=1.0)


# =============================================================================
# 2. Corners
# =============================================================================

def make_corners(n_samples: int = 2000, noise: float = 0.1,
                 random_state=None, radius: float = 0.75
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Corners dataset (Appendix E.1).

    Four quarter-circles of ``radius`` (default 0.75) are centred on the four
    corners ``(+/-1, +/-1)`` of the square ``[-1, 1]^2``. Points inside any of
    the four circular regions are class 0 (paper's -1); points outside all of
    them are class 1 (paper's +1). Returns an exactly balanced ``(X, y)``.
    """
    rng = _as_rng(random_state)
    corners = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)

    def label(x1, x2):
        pts = np.stack([x1, x2], axis=1)               # (B, 2)
        d = np.linalg.norm(pts[:, None, :] - corners[None, :, :], axis=2)  # (B, 4)
        inside = (d.min(axis=1) < radius)
        return (~inside).astype(int)                   # outside -> +1 -> 1

    return _balanced_by_rejection(label, n_samples, noise, rng, bound=1.0)


# =============================================================================
# 3. Spiral
# =============================================================================

def make_spiral(n_samples: int = 2000, noise: float = 0.1,
                random_state=None, n_turns: float = 2.0,
                max_radius: float = 0.95
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Two-spiral dataset (Appendix E.1).

    Two interleaved Archimedean spirals share the same radial profile but are
    offset by pi in angle: class 1 (paper's +1) spirals out counter-clockwise
    from the origin, class 0 (paper's -1) is its point reflection. The radius
    grows linearly from 0 to ``max_radius`` over ``n_turns`` revolutions. Built
    exactly balanced by construction (one arm per class). Gaussian jitter of
    standard deviation ``noise`` is added so the arms are not perfectly clean.
    """
    rng = _as_rng(random_state)
    n1 = n_samples // 2
    n0 = n_samples - n1

    def arm(n, phase):
        t = rng.uniform(0.0, 1.0, size=n)
        theta = 2.0 * np.pi * n_turns * t + phase
        r = max_radius * t
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y], axis=1)

    X0 = arm(n0, phase=np.pi)     # class 0  (paper -1)
    X1 = arm(n1, phase=0.0)       # class 1  (paper +1)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    if noise:
        X = X + rng.normal(0.0, noise, size=X.shape)

    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


# =============================================================================
# 4. Circles (annulus)
# =============================================================================

def make_circles(n_samples: int = 2000, noise: float = 0.1,
                 random_state=None) -> Tuple[np.ndarray, np.ndarray]:
    """Circles / annulus dataset (Appendix E.1).

    Two concentric circles of radii ``0.5 * sqrt(2/pi)`` and ``sqrt(2/pi)`` define
    an annular ring on the square ``[-1, 1]^2``. Points *inside the ring* are
    class 0 (paper's -1); points in the central disk or outside the outer circle
    are class 1 (paper's +1). Returns an exactly balanced ``(X, y)``.

    Note. This is the paper's annulus problem, not scikit-learn's
    ``sklearn.datasets.make_circles`` (two nested circles). The name follows the
    paper; import it explicitly to avoid confusion with the scikit-learn helper.
    """
    rng = _as_rng(random_state)
    r_outer = np.sqrt(2.0 / np.pi)        # ~0.798
    r_inner = 0.5 * r_outer               # ~0.399

    def label(x1, x2):
        r = np.sqrt(x1 ** 2 + x2 ** 2)
        in_ring = (r > r_inner) & (r < r_outer)
        return (~in_ring).astype(int)     # disk or exterior -> +1 -> 1

    return _balanced_by_rejection(label, n_samples, noise, rng, bound=1.0)


# =============================================================================
# Registry + dispatcher
# =============================================================================

DATASETS: Dict[str, Callable[..., Tuple[np.ndarray, np.ndarray]]] = {
    "sinus":   make_sinus,
    "corners": make_corners,
    "spiral":  make_spiral,
    "circles": make_circles,
}

# One-line human-readable description per dataset, for notebook headers/captions.
DATASET_INFO: Dict[str, str] = {
    "sinus":   "Points above/below f(x1) = -0.8 sin(pi x1).",
    "corners": "Four quarter-circles (r=0.75) at the corners of [-1, 1]^2.",
    "spiral":  "Two interleaved Archimedean spirals.",
    "circles": "Annular ring between radii 0.5*sqrt(2/pi) and sqrt(2/pi).",
}


def make_dataset(name: str, n_samples: int = 2000, noise: float = 0.1,
                 random_state=None) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to one of the four generators by name.

    name : 'sinus' | 'corners' | 'spiral' | 'circles'
    """
    if name not in DATASETS:
        raise ValueError(
            f"unknown dataset {name!r}; choose from {sorted(DATASETS)}"
        )
    return DATASETS[name](n_samples=n_samples, noise=noise,
                          random_state=random_state)
