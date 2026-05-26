"""
qml_core.py
===========
Shared quantum-classical core for the evaluation framework. Contains:

    Quantum primitives
    ------------------
    fast_vqc_fixed_states(w, X)        -> (B, 4) state batch
    fast_vqc_trainable_states(w, X)    -> (B, 4)
    fast_dr_states(w, X)               -> (B, 2)
    fast_fixed_encoding_states(X)      -> (B, 4)        (encoding only)
    fast_trainable_encoding_states(w, X) -> (B, 4)      (encoding only)
    fast_z0(states)                    -> (B,) <Z_0> in [-1, 1]
    fast_states(theta, X, circuit_type='fixed')   dispatch entry point
    fast_predict(theta, X, circuit_type='fixed')  dispatch entry point

    Bloch coordinates
    -----------------
    state_to_bloch_batch(states, qubit=0) -> (B, 3) Bloch coords
    state_to_bloch(state, qubit=0)        -> (3,)   convenience wrapper

    Optimisation
    ------------
    AdamOptimizer(learning_rate, beta1, beta2, epsilon)

    Training (parameter-shift)
    --------------------------
    parameter_shift_gradient(theta, X, y, circuit_type='fixed', sample_weight=None)
        -> (grads, train_loss, train_acc)
    cost_np(theta, X, y, circuit_type='fixed', sample_weight=None)
        -> (loss, acc, scores)
    train_ps_run(params_init, X_train, y_train, X_val, y_val,
                 circuit_type='fixed', n_iter=200, lr=0.015,
                 sample_weight=None, select_by='val_loss',
                 store_param_history=False, verbose=False) -> history dict
    train_ps_multi_seed(seeds, n_params, X_train, y_train, X_val, y_val,
                         circuit_type='fixed', init_fn=None,
                         n_iter=200, lr=0.015, sample_weight=None,
                         select_by='val_loss',
                         store_param_history=False, label='') -> (best, all_runs)

    Counters
    --------
    reset_counter()
    get_counter()

Conventions:
    * State ordering big-endian: |q0 q1> -> state[2*q0 + q1]
    * RY(theta) = exp(-i theta Y / 2) = [[c, -s], [s, c]],  c=cos(t/2), s=sin(t/2)
    * RZ(theta) = exp(-i theta Z / 2) = diag(e^{-it/2}, e^{+it/2})
    * CZ control=q0, target=q1 -> diag(1, 1, 1, -1)
    * Labels are in {-1, +1} for the squared loss on <Z_0>.

Pure NumPy. No external dependencies on Qibo, PyTorch, or scikit-learn.
The autodifferentiation pathway used in Case Study 1 of the thesis
(Qibo+PyTorch) lives in the notebook itself, not here.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# 1. Counter for circuit evaluations
# =============================================================================

_circuit_eval_counter: int = 0


def reset_counter() -> None:
    """Reset the per-process circuit-evaluation counter to zero."""
    global _circuit_eval_counter
    _circuit_eval_counter = 0


def get_counter() -> int:
    """Return the per-process circuit-evaluation counter."""
    return _circuit_eval_counter


def bump_counter(n: int = 1) -> None:
    """Increment the circuit-evaluation counter by `n`.

    Used by external code paths (e.g. the Qibo+PyTorch autodifferentiation
    pathway in Case Study 1) that want to share a single eval-counting
    source of truth with the parameter-shift training path defined in this
    module.
    """
    global _circuit_eval_counter
    _circuit_eval_counter += n


# =============================================================================
# 2. Single-qubit gate factories (scalar OR batched theta)
# =============================================================================

def _ry_batch(theta):
    """RY rotation matrix. theta: scalar or (B,). Returns (2,2) or (B, 2, 2)."""
    theta = np.asarray(theta)
    c = np.cos(theta * 0.5)
    s = np.sin(theta * 0.5)
    if theta.ndim == 0:
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    g = np.empty(theta.shape + (2, 2), dtype=np.complex128)
    g[..., 0, 0] = c
    g[..., 0, 1] = -s
    g[..., 1, 0] = s
    g[..., 1, 1] = c
    return g


def _rz_batch(theta):
    """RZ rotation matrix. theta: scalar or (B,). Returns (2,2) or (B, 2, 2)."""
    theta = np.asarray(theta)
    if theta.ndim == 0:
        return np.array([[np.exp(-0.5j * theta), 0.0],
                         [0.0, np.exp(0.5j * theta)]], dtype=np.complex128)
    g = np.zeros(theta.shape + (2, 2), dtype=np.complex128)
    g[..., 0, 0] = np.exp(-0.5j * theta)
    g[..., 1, 1] = np.exp(0.5j * theta)
    return g


# CZ as a diagonal vector applied via element-wise multiply on (B, 4) states.
_CZ_DIAG = np.array([1, 1, 1, -1], dtype=np.complex128)


# =============================================================================
# 3. Gate application on batched states (B, 4)
# =============================================================================

def _apply_q0(states, g):
    """Apply 2x2 gate g to qubit 0. states: (B, 4). g: (2,2) or (B, 2, 2)."""
    s = states.reshape(states.shape[:-1] + (2, 2))
    if g.ndim == 2:
        out = np.einsum('ij,...jk->...ik', g, s)
    else:
        out = np.einsum('bij,bjk->bik', g, s)
    return out.reshape(states.shape)


def _apply_q1(states, g):
    """Apply 2x2 gate g to qubit 1. states: (B, 4). g: (2,2) or (B, 2, 2)."""
    s = states.reshape(states.shape[:-1] + (2, 2))
    if g.ndim == 2:
        out = np.einsum('...ij,kj->...ik', s, g)
    else:
        out = np.einsum('bij,bkj->bik', s, g)
    return out.reshape(states.shape)


# =============================================================================
# 4. Full-circuit forwards for each architecture
# =============================================================================

def fast_vqc_fixed_states(w, X):
    """Fixed-encoding VQC.

    w:  (18,)  or (B, 18)  parameters
    X:  (B, 2)              features
    Returns (B, 4) final state vectors before measurement.
    """
    B = X.shape[0]
    s = np.zeros((B, 4), dtype=np.complex128)
    s[:, 0] = 1.0
    # Encoding (no trainable parameters)
    s = _apply_q0(s, _ry_batch(X[:, 0]))
    s = _apply_q1(s, _ry_batch(X[:, 0]))
    s = _apply_q0(s, _rz_batch(X[:, 1]))
    s = _apply_q1(s, _rz_batch(X[:, 1]))
    # 4 variational layers
    for i in range(4):
        j = i * 4
        s = _apply_q0(s, _ry_batch(w[..., j]))
        s = _apply_q1(s, _ry_batch(w[..., j + 1]))
        s = s * _CZ_DIAG
        s = _apply_q0(s, _ry_batch(w[..., j + 2]))
        s = _apply_q1(s, _ry_batch(w[..., j + 3]))
    # Final rotation
    s = _apply_q0(s, _ry_batch(w[..., 16]))
    s = _apply_q1(s, _ry_batch(w[..., 17]))
    return s


def fast_vqc_trainable_states(w, X):
    """Trainable-encoding VQC.

    w:  (22,)  or (B, 22)  parameters; w[0..3] are encoding weights.
    X:  (B, 2)              features
    Returns (B, 4) final state vectors before measurement.
    """
    B = X.shape[0]
    s = np.zeros((B, 4), dtype=np.complex128)
    s[:, 0] = 1.0
    # Trainable encoding (theta = w_k * x_j)
    s = _apply_q0(s, _ry_batch(w[..., 0] * X[:, 0]))
    s = _apply_q1(s, _ry_batch(w[..., 1] * X[:, 0]))
    s = _apply_q0(s, _rz_batch(w[..., 2] * X[:, 1]))
    s = _apply_q1(s, _rz_batch(w[..., 3] * X[:, 1]))
    # 4 variational layers
    for i in range(4):
        j = 4 + i * 4
        s = _apply_q0(s, _ry_batch(w[..., j]))
        s = _apply_q1(s, _ry_batch(w[..., j + 1]))
        s = s * _CZ_DIAG
        s = _apply_q0(s, _ry_batch(w[..., j + 2]))
        s = _apply_q1(s, _ry_batch(w[..., j + 3]))
    # Final rotation
    s = _apply_q0(s, _ry_batch(w[..., 20]))
    s = _apply_q1(s, _ry_batch(w[..., 21]))
    return s


def fast_dr_states(w, X):
    """Data Reuploading. Single qubit, three layers, additive encoding.

    w: (6,)  or (B, 6)  parameters
    X: (B, 2)            features
    Returns (B, 2) final state vectors (one qubit) before measurement.
    """
    B = X.shape[0]
    s = np.zeros((B, 2), dtype=np.complex128)
    s[:, 0] = 1.0
    for i in range(3):
        gate = _ry_batch(X[:, 0] + w[..., 2 * i])
        s = np.einsum('bij,bj->bi', gate, s)
        gate = _rz_batch(X[:, 1] + w[..., 2 * i + 1])
        s = np.einsum('bij,bj->bi', gate, s)
    return s


# =============================================================================
# 5. Encoding-only forwards (used by the post-encoding Bloch diagnostic)
# =============================================================================

def fast_fixed_encoding_states(X):
    """Encoding block of the Fixed VQC, no variational layers.

    X: (B, 2). Returns (B, 4). No trainable parameters.
    """
    B = X.shape[0]
    s = np.zeros((B, 4), dtype=np.complex128)
    s[:, 0] = 1.0
    s = _apply_q0(s, _ry_batch(X[:, 0]))
    s = _apply_q1(s, _ry_batch(X[:, 0]))
    s = _apply_q0(s, _rz_batch(X[:, 1]))
    s = _apply_q1(s, _rz_batch(X[:, 1]))
    return s


def fast_trainable_encoding_states(w, X):
    """Encoding block of the Trainable VQC, no variational layers.

    w: (22,) or (B, 22)  parameters (only w[0..3] are used here)
    X: (B, 2). Returns (B, 4).
    """
    B = X.shape[0]
    s = np.zeros((B, 4), dtype=np.complex128)
    s[:, 0] = 1.0
    s = _apply_q0(s, _ry_batch(w[..., 0] * X[:, 0]))
    s = _apply_q1(s, _ry_batch(w[..., 1] * X[:, 0]))
    s = _apply_q0(s, _rz_batch(w[..., 2] * X[:, 1]))
    s = _apply_q1(s, _rz_batch(w[..., 3] * X[:, 1]))
    return s


# =============================================================================
# 6. Observable measurement: <Z_0>
# =============================================================================

def fast_z0(states):
    """<Z_0> for a batch of states.

    states: (B, 2) for 1-qubit or (B, 4) for 2-qubit.
    Returns (B,) real array in [-1, 1].
    """
    n = states.shape[-1]
    half = n // 2
    p = states.real ** 2 + states.imag ** 2
    return p[..., :half].sum(axis=-1) - p[..., half:].sum(axis=-1)


# =============================================================================
# 7. Single dispatch entry point
# =============================================================================

def fast_states(theta, X, circuit_type: str = 'fixed') -> np.ndarray:
    """Dispatch to the appropriate full-circuit forward.

    circuit_type: 'fixed' | 'trainable' | 'dr' | 'fixed_encoding' | 'trainable_encoding'
    """
    if circuit_type == 'dr':
        return fast_dr_states(theta, X)
    if circuit_type == 'trainable':
        return fast_vqc_trainable_states(theta, X)
    if circuit_type == 'fixed_encoding':
        return fast_fixed_encoding_states(X)
    if circuit_type == 'trainable_encoding':
        return fast_trainable_encoding_states(theta, X)
    return fast_vqc_fixed_states(theta, X)


def fast_predict(theta, X, circuit_type: str = 'fixed') -> np.ndarray:
    """Continuous prediction <Z_0> in [-1, 1] for a batch of inputs."""
    return fast_z0(fast_states(theta, X, circuit_type=circuit_type)).real


# =============================================================================
# 8. Bloch sphere coordinates
# =============================================================================

def state_to_bloch_batch(states, qubit: int = 0) -> np.ndarray:
    """Bloch vector of qubit `qubit` for a batch of pure states.

    states: (B, 2) for 1-qubit, (B, 4) for 2-qubit.
    Returns (B, 3) array of (rx, ry, rz) Bloch coordinates.

    For 2-qubit input, the partial trace over the other qubit is computed
    automatically. The resulting reduced density matrix is converted to a
    Bloch vector via the standard formula
        rx = 2 Re(rho_01),  ry = 2 Im(rho_10),  rz = rho_00 - rho_11.
    """
    if states.shape[-1] == 2:
        rho_00 = (states[:, 0] * states[:, 0].conj()).real
        rho_11 = (states[:, 1] * states[:, 1].conj()).real
        rho_01 = states[:, 0] * states[:, 1].conj()
        rho_10 = states[:, 1] * states[:, 0].conj()
    elif states.shape[-1] == 4:
        psi = states.reshape(-1, 2, 2)               # (B, q0, q1)
        if qubit == 0:
            rho = np.einsum('bij,bkj->bik', psi, psi.conj())  # trace out q1
        else:
            rho = np.einsum('bji,bjk->bik', psi, psi.conj())  # trace out q0
        rho_00 = rho[:, 0, 0].real
        rho_11 = rho[:, 1, 1].real
        rho_01 = rho[:, 0, 1]
        rho_10 = rho[:, 1, 0]
    else:
        raise ValueError(f"Unexpected state size {states.shape[-1]}")
    rx = 2 * rho_01.real
    ry = 2 * rho_10.imag
    rz = rho_00 - rho_11
    return np.stack([rx, ry, rz], axis=-1)


def state_to_bloch(state_vec, qubit: int = 0) -> Tuple[float, float, float]:
    """Convenience wrapper around `state_to_bloch_batch` for a single state."""
    sv = np.asarray(state_vec, dtype=complex).reshape(1, -1)
    return tuple(state_to_bloch_batch(sv, qubit=qubit)[0])


# =============================================================================
# 9. Adam optimiser (numpy)
# =============================================================================

class AdamOptimizer:
    """Adam optimiser in pure NumPy.

    Shared by the autodifferentiation and parameter-shift training paths so
    that the only methodological difference between them is how the gradient
    is computed.
    """

    def __init__(self, learning_rate: float = 0.015,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t: int = 0

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# =============================================================================
# 10. Cost function (vectorised, with optional sample weights)
# =============================================================================

def cost_np(theta: np.ndarray,
            X: np.ndarray,
            y: np.ndarray,
            circuit_type: str = 'fixed',
            sample_weight: Optional[np.ndarray] = None
            ) -> Tuple[float, float, np.ndarray]:
    """Forward-only weighted MSE cost.

    Returns (loss, acc, scores). Bumps the global circuit-eval counter by N.
    Labels are expected in {-1, +1}.
    """
    global _circuit_eval_counter
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    scores = fast_predict(np.asarray(theta), X, circuit_type=circuit_type)
    _circuit_eval_counter += len(X)
    if sample_weight is None:
        sw = np.ones(len(X), dtype=float)
    else:
        sw = np.asarray(sample_weight, dtype=float)
    loss = float(np.sum(sw * (scores - y) ** 2) / np.sum(sw))
    acc = float(np.mean(np.sign(scores) == np.sign(y)))
    return loss, acc, scores


# =============================================================================
# 11. Parameter-shift gradient (vectorised, with optional sample weights)
# =============================================================================

# Trainable-encoding parameters w[0..3] map to feature indices.
_ENCODING_FEATURE_MAP = {0: 0, 1: 0, 2: 1, 3: 1}
_N_ENCODING_PARAMS = 4


def parameter_shift_gradient(theta: np.ndarray,
                             X: np.ndarray,
                             y: np.ndarray,
                             circuit_type: str = 'fixed',
                             sample_weight: Optional[np.ndarray] = None
                             ) -> Tuple[np.ndarray, float, float]:
    """Vectorised parameter-shift gradient with optional sample weights.

    For each parameter k two batched forwards are run (theta_plus and
    theta_minus over all N samples at once) instead of 2*N sequential
    forwards. For trainable-encoding parameters (multiplicative w*x) the
    shift is per-sample and equal to pi / (2 x_j); the same vectorisation
    pattern applies because numpy broadcasts the per-sample shifts cleanly.

    Counter accounting matches the original sequential semantics: each
    batched forward over N samples adds N to the eval counter, so the
    per-iteration count is the same as before vectorisation and the
    eval-count comparison with the autodifferentiation path stays
    meaningful.

    Labels are expected in {-1, +1}.

    Returns (gradients, train_loss, train_acc).
    """
    global _circuit_eval_counter
    theta = np.asarray(theta, dtype=float)
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    n_params = len(theta)
    N = X.shape[0]

    if sample_weight is None:
        sw = np.ones(N, dtype=float)
    else:
        sw = np.asarray(sample_weight, dtype=float)
        if sw.shape[0] != N:
            raise ValueError("sample_weight has the wrong length")
    sw_sum = float(np.sum(sw))

    f_current = fast_predict(theta, X, circuit_type=circuit_type)
    _circuit_eval_counter += N
    residuals = f_current - y
    train_loss = float(np.sum(sw * residuals ** 2) / sw_sum)
    train_acc = float(np.mean(np.sign(f_current) == np.sign(y)))

    gradients = np.zeros(n_params)
    for k in range(n_params):
        is_encoding = (circuit_type == 'trainable') and (k < _N_ENCODING_PARAMS)
        if is_encoding:
            feat_idx = _ENCODING_FEATURE_MAP[k]
            x_j = X[:, feat_idx].astype(float)
            mask = np.abs(x_j) >= 1e-10
            shifts = np.zeros(N)
            shifts[mask] = np.pi / (2.0 * x_j[mask])
            theta_batch_plus = np.broadcast_to(theta, (N, n_params)).copy()
            theta_batch_minus = np.broadcast_to(theta, (N, n_params)).copy()
            theta_batch_plus[:, k] += shifts
            theta_batch_minus[:, k] -= shifts
            f_plus = fast_predict(theta_batch_plus, X, circuit_type=circuit_type)
            f_minus = fast_predict(theta_batch_minus, X, circuit_type=circuit_type)
            df_dw = x_j * (f_plus - f_minus) / 2.0
            df_dw[~mask] = 0.0
        else:
            theta_plus = theta.copy()
            theta_plus[k] += np.pi / 2.0
            theta_minus = theta.copy()
            theta_minus[k] -= np.pi / 2.0
            f_plus = fast_predict(theta_plus, X, circuit_type=circuit_type)
            f_minus = fast_predict(theta_minus, X, circuit_type=circuit_type)
            df_dw = (f_plus - f_minus) / 2.0
        _circuit_eval_counter += 2 * N
        # Weighted gradient: d/dw [sum_i sw_i (f_i - y_i)^2 / sw_sum]
        gradients[k] = float(np.sum(sw * 2.0 * residuals * df_dw) / sw_sum)
    return gradients, train_loss, train_acc


# =============================================================================
# 12. Single parameter-shift training run
# =============================================================================

def train_ps_run(params_init: np.ndarray,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 circuit_type: str = 'fixed',
                 n_iter: int = 200,
                 lr: float = 0.015,
                 sample_weight: Optional[np.ndarray] = None,
                 select_by: str = 'val_loss',
                 store_param_history: bool = False,
                 verbose: bool = False) -> Dict:
    """Single parameter-shift training run.

    Parameters
    ----------
    params_init : (n_params,) initial parameter vector
    X_train, y_train, X_val, y_val : training / validation arrays.
        Labels are in {-1, +1}.
    circuit_type : 'fixed' | 'trainable' | 'dr'
    n_iter, lr   : standard Adam hyperparameters
    sample_weight: optional per-training-sample weight (None = uniform)
    select_by    : 'val_loss' (default, recommended for imbalanced setting)
                   or 'val_acc' (with tiebreak on lower val_loss).
    store_param_history : if True, history['params'] keeps the parameter
                          vector at every iteration. Required by the Bloch
                          slider in Case Study 1; off by default to save
                          memory.
    verbose      : print a one-line summary per iteration

    Returns
    -------
    history dict with keys:
        'train_loss', 'train_acc', 'val_loss', 'val_acc', 'grad_norms'  (lists)
        'params' (list, only if store_param_history=True)
        'best_val_loss', 'best_val_acc', 'best_val_iter', 'best_val_params'
        'final_params'
    """
    if select_by not in ('val_loss', 'val_acc'):
        raise ValueError(f"select_by must be 'val_loss' or 'val_acc', got {select_by!r}")

    params = np.asarray(params_init, dtype=float).copy()
    optimizer = AdamOptimizer(learning_rate=lr)
    h: Dict = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'grad_norms': [],
    }
    if store_param_history:
        h['params'] = []

    best_vl = np.inf
    best_va = -np.inf
    best_iter = -1
    best_params = params.copy()

    for it in range(n_iter):
        if store_param_history:
            h['params'].append(params.copy())
        grads, tl, ta = parameter_shift_gradient(
            params, X_train, y_train,
            circuit_type=circuit_type, sample_weight=sample_weight,
        )
        gn = float(np.linalg.norm(grads))
        h['grad_norms'].append(gn)
        vl, va, _ = cost_np(params, X_val, y_val, circuit_type=circuit_type)
        h['train_loss'].append(float(tl))
        h['train_acc'].append(float(ta))
        h['val_loss'].append(float(vl))
        h['val_acc'].append(float(va))

        is_better = (
            (select_by == 'val_loss' and vl < best_vl) or
            (select_by == 'val_acc'  and (va > best_va or (va == best_va and vl < best_vl)))
        )
        if is_better:
            best_vl = vl
            best_va = va
            best_iter = it
            best_params = params.copy()

        params = optimizer.step(params, grads)

        if verbose:
            print(f'  iter {it:3d}: train {tl:.4f}/{ta:.4f}  '
                  f'val {vl:.4f}/{va:.4f}  ||g||={gn:.6f}')

    if store_param_history:
        h['params'].append(params.copy())

    h['best_val_loss'] = best_vl
    h['best_val_acc'] = best_va
    h['best_val_iter'] = best_iter
    h['best_val_params'] = best_params
    h['final_params'] = params.copy()
    return h


# =============================================================================
# 13. Multi-seed runner
# =============================================================================

def train_ps_multi_seed(seeds: List[int],
                         n_params: int,
                         X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         circuit_type: str = 'fixed',
                         init_fn: Optional[Callable[[int, int], np.ndarray]] = None,
                         n_iter: int = 200,
                         lr: float = 0.015,
                         sample_weight: Optional[np.ndarray] = None,
                         select_by: str = 'val_loss',
                         store_param_history: bool = False,
                         label: str = '',
                         ) -> Tuple[Dict, List[Dict]]:
    """Run train_ps_run over several seeds and pick the best.

    init_fn : callable(seed, n_params) -> initial parameter vector.
              Default: pi * Uniform[0, 1]^n_params.
    select_by : ranking criterion across seeds (matches the within-run one).

    Returns (best_run, all_runs).
    """
    if init_fn is None:
        def init_fn(seed: int, n: int) -> np.ndarray:
            rng = np.random.RandomState(seed)
            return np.pi * rng.rand(n)

    label_str = f' [{label}]' if label else ''
    print(f"Multi-seed PS training{label_str}: "
          f"{len(seeds)} seeds x {n_iter} iter ({circuit_type})")
    all_runs: List[Dict] = []
    for seed in seeds:
        params_init = init_fn(seed, n_params)
        reset_counter()
        t0 = time.time()
        h = train_ps_run(
            params_init, X_train, y_train, X_val, y_val,
            circuit_type=circuit_type,
            n_iter=n_iter, lr=lr,
            sample_weight=sample_weight,
            select_by=select_by,
            store_param_history=store_param_history,
        )
        h['time_s'] = time.time() - t0
        h['evals'] = get_counter()
        h['seed'] = seed
        all_runs.append(h)
        print(f"  seed={seed}: final val_loss={h['val_loss'][-1]:.4f} "
              f"val_acc={h['val_acc'][-1]:.4f}  "
              f"best ({select_by})={h['best_val_loss' if select_by=='val_loss' else 'best_val_acc']:.4f} "
              f"(it {h['best_val_iter']:>3})  time={h['time_s']:.1f}s")

    # Pick best across seeds, matching the per-run criterion
    if select_by == 'val_loss':
        best_idx = min(range(len(all_runs)),
                       key=lambda i: all_runs[i]['best_val_loss'])
    else:
        best_idx = max(range(len(all_runs)),
                       key=lambda i: (all_runs[i]['best_val_acc'],
                                      -all_runs[i]['best_val_loss']))
    best = all_runs[best_idx]
    print(f"-> Best: seed={best['seed']}  best_{select_by}="
          f"{best['best_val_loss' if select_by=='val_loss' else 'best_val_acc']:.4f}")
    return best, all_runs
