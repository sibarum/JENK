"""Path-B experiment with full transparency: show every piece.

Pulls back the curtain on tmp/path_b_synthetic_recovery.py — prints the
symbolic forward/loss/grad expressions, sample data points, training
trajectory at log-spaced steps, and the recovered baseline matrix.

Run from JENK/:
    PYTHONPATH=. conda run -n traction python tmp/path_b_detail.py
"""

from __future__ import annotations

import time

import numpy as np
import sympy as sp

from jenk.symbolic_neuron import SymbolicNeuron


# Same hyperparameters as the main experiment, for reproducibility.
N_SAMPLES = 200
N_STEPS = 500
LR = 0.05
S_INIT = 0.0
SEED = 42


def _print_symbolic_setup():
    print('=' * 92)
    print(' STEP 1 — Symbolic setup: what the trainable network looks like algebraically')
    print('=' * 92)
    s = sp.Symbol('s', real=True)
    xa, xb = sp.symbols('xa xb', real=True)
    ta, tb = sp.symbols('ta tb', real=True)

    neuron = SymbolicNeuron.generator(s)
    print(f'  Neuron: SymbolicNeuron.generator(s)   (c=0, d=1, no bias)')
    print(f'  Matrix M(s) = {neuron.matrix().tolist()}')
    pred_a, pred_b = neuron.forward(xa, xb)
    print(f'  Forward:')
    print(f'    pred_a(s, xa, xb) = {pred_a}')
    print(f'    pred_b(s, xa, xb) = {pred_b}')

    sample_loss = (pred_a - ta) ** 2 + (pred_b - tb) ** 2
    sample_loss_expanded = sp.expand(sample_loss)
    print(f'  Per-sample loss = (pred_a - ta)^2 + (pred_b - tb)^2')
    print(f'                  = {sample_loss_expanded}')
    sample_grad = sp.diff(sample_loss, s)
    print(f'  d/ds (per-sample loss) = {sp.simplify(sample_grad)}')
    print()
    return s, xa, xb, ta, tb, sample_loss, sample_grad


def _print_data_sample(s_true: float, X: np.ndarray, Y: np.ndarray, n_show: int = 5):
    print('=' * 92)
    print(f' STEP 2 — Data: Y = M(s_true={s_true}) X     (showing first {n_show} of {len(X)})')
    print('=' * 92)
    M_true = np.array([[0.0, -1.0], [1.0, s_true]])
    print(f'  True M(s_true={s_true}) =')
    print(f'    {M_true.tolist()}')
    print(f'  First {n_show} samples:')
    print(f"    {'X (input)':<28}    {'Y (target)':<28}")
    for i in range(n_show):
        print(f'    ({X[i, 0]:+.4f}, {X[i, 1]:+.4f})        ({Y[i, 0]:+.4f}, {Y[i, 1]:+.4f})')
    print()


def _train_with_trajectory(loss_fn, grad_fn, X: np.ndarray, Y: np.ndarray):
    """Standard SGD over s, but record (step, s, loss) at log-spaced ticks."""
    log_steps = sorted(set(
        [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        + [N_STEPS - 1]
    ))
    s_current = float(S_INIT)
    snapshots = []
    xa, xb = X[:, 0], X[:, 1]
    ta, tb = Y[:, 0], Y[:, 1]
    for step in range(N_STEPS):
        per_sample_loss = loss_fn(s_current, xa, xb, ta, tb)
        per_sample_grad = grad_fn(s_current, xa, xb, ta, tb)
        loss = float(np.mean(per_sample_loss))
        grad = float(np.mean(per_sample_grad))
        if step in log_steps:
            snapshots.append((step, s_current, loss, grad))
        s_current = s_current - LR * grad
    return s_current, snapshots


def _train_linear_with_trajectory(X: np.ndarray, Y: np.ndarray, rng):
    log_steps = sorted(set(
        [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        + [N_STEPS - 1]
    ))
    W = rng.normal(size=(2, 2)) * 0.1
    snapshots = []
    n = X.shape[0]
    for step in range(N_STEPS):
        pred = X @ W.T
        err = pred - Y
        loss = float(np.mean(err ** 2))
        if step in log_steps:
            snapshots.append((step, W.copy(), loss))
        grad_W = (2.0 / n) * err.T @ X
        W = W - LR * grad_W
    return W, snapshots


def _print_symbolic_trajectory(s_true: float, snapshots, final_s: float):
    print('=' * 92)
    print(f' STEP 3 — Training trajectory of SymbolicNeuron (init s={S_INIT}, lr={LR})')
    print('=' * 92)
    print(f"  {'step':>5}  {'s':>13}  {'|s - s_true|':>14}  {'loss':>13}  {'grad':>13}")
    print('  ' + '-' * 76)
    for step, s, loss, grad in snapshots:
        print(f'  {step:>5}  {s:>+13.10f}  {abs(s - s_true):>14.3e}  {loss:>13.3e}  {grad:>+13.3e}')
    print(f"  Final after {N_STEPS} steps: s = {final_s:+.15f}   (true s* = {s_true})")
    print()


def _print_linear_trajectory(s_true: float, snapshots, final_W: np.ndarray):
    M_true = np.array([[0.0, -1.0], [1.0, s_true]])
    print('=' * 92)
    print(f' STEP 4 — Training trajectory of free 2x2 baseline (4 params)')
    print('=' * 92)
    print(f"  {'step':>5}  {'W (flattened, [m00 m01 m10 m11])':<46}  {'loss':>13}  {'||W-M||_F':>11}")
    print('  ' + '-' * 90)
    for step, W, loss in snapshots:
        flat = ' '.join(f'{v:+.4f}' for v in W.ravel())
        err = float(np.linalg.norm(W - M_true, 'fro'))
        print(f'  {step:>5}  [{flat}]  {loss:>13.3e}  {err:>11.3e}')
    print(f'  Final W after {N_STEPS} steps:')
    print(f'    {final_W.tolist()}')
    print(f'  True M(s_true={s_true}):')
    print(f'    {M_true.tolist()}')
    print(f'  ||W - M||_F = {float(np.linalg.norm(final_W - M_true, "fro")):.3e}')
    print()


def run_one_detailed(s_true: float, label: str, loss_fn, grad_fn):
    print()
    print('#' * 92)
    print(f'#  CASE: s_true = {s_true}  ({label})')
    print('#' * 92)

    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(N_SAMPLES, 2))
    M_true = np.array([[0.0, -1.0], [1.0, s_true]])
    Y = X @ M_true.T

    _print_data_sample(s_true, X, Y)

    t0 = time.perf_counter()
    final_s, sym_snapshots = _train_with_trajectory(loss_fn, grad_fn, X, Y)
    sym_time = time.perf_counter() - t0
    _print_symbolic_trajectory(s_true, sym_snapshots, final_s)
    print(f'  Wall-clock: {sym_time * 1000:.1f} ms for {N_STEPS} steps')
    print()

    rng_b = np.random.default_rng(SEED)
    rng_b.normal(size=(N_SAMPLES, 2))  # consume the data draw to match seed offset
    t0 = time.perf_counter()
    final_W, lin_snapshots = _train_linear_with_trajectory(X, Y, rng_b)
    lin_time = time.perf_counter() - t0
    _print_linear_trajectory(s_true, lin_snapshots, final_W)
    print(f'  Wall-clock: {lin_time * 1000:.1f} ms for {N_STEPS} steps')


def main():
    _print_symbolic_setup()
    s = sp.Symbol('s', real=True)
    xa, xb = sp.symbols('xa xb', real=True)
    ta, tb = sp.symbols('ta tb', real=True)
    neuron = SymbolicNeuron.generator(s)
    pred_a, pred_b = neuron.forward(xa, xb)
    sample_loss = (pred_a - ta) ** 2 + (pred_b - tb) ** 2
    sample_grad = sp.diff(sample_loss, s)
    loss_fn = sp.lambdify((s, xa, xb, ta, tb), sample_loss, 'numpy')
    grad_fn = sp.lambdify((s, xa, xb, ta, tb), sample_grad, 'numpy')

    # Two representative cases: one elliptic interior, one hyperbolic exterior.
    run_one_detailed(0.5, 'elliptic interior, |s| < 2', loss_fn, grad_fn)
    run_one_detailed(2.5, 'hyperbolic exterior, |s| > 2', loss_fn, grad_fn)


if __name__ == '__main__':
    main()
