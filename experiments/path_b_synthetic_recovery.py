"""Path-B synthetic recovery experiment.

Test whether a SymbolicNeuron can be trained (via symbolic-gradient SGD)
to recover a known ground-truth trace s* from data generated as
Y = M(s*) X, where M(s*) = [[0, -1], [1, s*]].

Compared against a free 4-parameter 2x2 linear layer trained with the
same SGD on the same data. The free linear layer is a strictly more
expressive model (it can fit any 2x2 transform), so the question is
whether the SymbolicNeuron's 1-parameter constrained model recovers the
ground truth and how it compares.

Run from JENK/:
    PYTHONPATH=. conda run -n traction python tmp/path_b_synthetic_recovery.py
"""

from __future__ import annotations

import time

import numpy as np
import sympy as sp

from jenk.graded import omega
from jenk.symbolic_neuron import SymbolicNeuron


# --- Data generation -----------------------------------------------------

def generate_data(s_true: float, n_samples: int, rng: np.random.Generator):
    """Generate (X, Y) where Y = M(s_true) X^T.

    X is shape (n_samples, 2), Y is shape (n_samples, 2). Inputs are
    standard normal.
    """
    X = rng.normal(size=(n_samples, 2))
    M = np.array([[0.0, -1.0], [1.0, s_true]])
    Y = X @ M.T
    return X, Y


# --- Symbolic-neuron training -------------------------------------------

def build_symbolic_loss_and_grad():
    """Compile the per-sample loss and dL/ds for a generator-weight neuron.

    Returns (loss_fn, grad_fn), each taking (s, xa, xb, ta, tb) and
    returning numpy arrays. They are vectorized over the sample axis.
    """
    s = sp.Symbol('s', real=True)
    xa, xb = sp.symbols('xa xb', real=True)
    ta, tb = sp.symbols('ta tb', real=True)

    neuron = SymbolicNeuron.generator(s)
    pred_a, pred_b = neuron.forward(xa, xb)
    sample_loss = (pred_a - ta) ** 2 + (pred_b - tb) ** 2
    sample_grad = sp.diff(sample_loss, s)

    loss_fn = sp.lambdify((s, xa, xb, ta, tb), sample_loss, 'numpy')
    grad_fn = sp.lambdify((s, xa, xb, ta, tb), sample_grad, 'numpy')
    return loss_fn, grad_fn


def train_symbolic_neuron(
    X, Y, s_init: float, lr: float, n_steps: int,
    loss_fn, grad_fn,
):
    """Gradient descent over a single learnable parameter `s`.

    Returns (s_history, loss_history). Vectorized over batch.
    """
    s_current = float(s_init)
    s_hist = [s_current]
    loss_hist = []
    xa, xb = X[:, 0], X[:, 1]
    ta, tb = Y[:, 0], Y[:, 1]
    for _ in range(n_steps):
        per_sample_loss = loss_fn(s_current, xa, xb, ta, tb)
        per_sample_grad = grad_fn(s_current, xa, xb, ta, tb)
        loss = float(np.mean(per_sample_loss))
        grad = float(np.mean(per_sample_grad))
        loss_hist.append(loss)
        s_current = s_current - lr * grad
        s_hist.append(s_current)
    return s_hist, loss_hist


# --- Linear baseline (4-parameter 2x2 matrix) ---------------------------

def train_linear_baseline(X, Y, lr: float, n_steps: int, rng: np.random.Generator):
    """Train a free 2x2 matrix W (no bias) via numpy SGD.

    Loss: mean((X @ W.T - Y)^2). Returns (W_history, loss_history).
    """
    W = rng.normal(size=(2, 2)) * 0.1  # small init
    W_hist = [W.copy()]
    loss_hist = []
    n = X.shape[0]
    for _ in range(n_steps):
        pred = X @ W.T  # (n, 2)
        err = pred - Y  # (n, 2)
        loss = float(np.mean(err ** 2))
        loss_hist.append(loss)
        # dL/dW = (2/n) * err.T @ X     (each entry of W contributes to a row of pred)
        grad_W = (2.0 / n) * err.T @ X
        W = W - lr * grad_W
        W_hist.append(W.copy())
    return W_hist, loss_hist


# --- Experiment runner --------------------------------------------------

S_TRUE_VALUES = [
    (0.0,  'hyperbolic-mode cardinal (j)'),
    (1.0,  'parabolic-mode cardinal (e)'),
    (-1.0, 'projective-mode cardinal (k)'),
    (0.5,  'mid-elliptic |s| < 2'),
    (1.7,  'near elliptic boundary'),
    (2.0,  'parabolic boundary |s| = 2'),
    (2.5,  'mild hyperbolic |s| > 2'),
    (3.0,  'deeper hyperbolic'),
]
N_SAMPLES = 200
N_STEPS = 500
LR = 0.05
S_INIT = 0.0
SEED = 42


def run_one(s_true: float, label: str, loss_fn, grad_fn):
    rng = np.random.default_rng(SEED)
    X, Y = generate_data(s_true, N_SAMPLES, rng)

    t0 = time.perf_counter()
    s_hist, sym_losses = train_symbolic_neuron(
        X, Y, S_INIT, LR, N_STEPS, loss_fn, grad_fn,
    )
    sym_time = time.perf_counter() - t0
    s_recovered = s_hist[-1]
    sym_final_loss = sym_losses[-1]

    rng_b = np.random.default_rng(SEED)  # same seed for baseline init
    t0 = time.perf_counter()
    W_hist, lin_losses = train_linear_baseline(X, Y, LR, N_STEPS, rng_b)
    lin_time = time.perf_counter() - t0
    W_recovered = W_hist[-1]
    lin_final_loss = lin_losses[-1]

    M_true = np.array([[0.0, -1.0], [1.0, s_true]])
    W_err_frob = float(np.linalg.norm(W_recovered - M_true, 'fro'))

    return {
        's_true': s_true,
        'label': label,
        's_recovered': s_recovered,
        's_err': abs(s_recovered - s_true),
        'sym_final_loss': sym_final_loss,
        'sym_time': sym_time,
        'W_recovered': W_recovered,
        'W_err_frob': W_err_frob,
        'lin_final_loss': lin_final_loss,
        'lin_time': lin_time,
    }


def print_results(results):
    print()
    print('=' * 92)
    print(' Synthetic-recovery results: Y = M(s_true) X, train SymbolicNeuron and free 2x2 baseline')
    print('=' * 92)
    header = f"{'s_true':>7}  {'label':<32}  {'recov s':>9}  {'|err|':>8}  {'sym loss':>11}  {'lin loss':>11}"
    print(header)
    print('-' * len(header))
    for r in results:
        print(
            f"{r['s_true']:>7.3f}  {r['label']:<32}  "
            f"{r['s_recovered']:>9.5f}  {r['s_err']:>8.2e}  "
            f"{r['sym_final_loss']:>11.3e}  {r['lin_final_loss']:>11.3e}"
        )
    print()
    print('Parameter counts: SymbolicNeuron = 1   |   Free 2x2 linear = 4')
    print('Steps: {}   |   LR: {}   |   N samples: {}   |   init s: {}'.format(
        N_STEPS, LR, N_SAMPLES, S_INIT,
    ))


# --- Probe: symbolic loss when s_true is omega --------------------------

def omega_probe():
    """Generate one symbolic data point with s_true = omega; show the
    symbolic loss as a function of the learnable s.

    We can't gradient-descend toward omega (it isn't a finite real), but
    we can read off the structure of the loss landscape symbolically.
    """
    print()
    print('=' * 92)
    print(' OMEGA PROBE: data generated by M(omega), inspect loss as function of trainable s')
    print('=' * 92)
    s = sp.Symbol('s', real=True)
    xa, xb, = sp.Symbol('xa', real=True), sp.Symbol('xb', real=True)

    # True transformation at s = omega:  M(omega) [xa, xb]^T = [-xb, xa + omega*xb]
    target_a = -xb
    target_b = xa + omega * xb

    # Trainable neuron at parameter s (a real symbol):
    neuron = SymbolicNeuron.generator(s)
    pred_a, pred_b = neuron.forward(xa, xb)

    sample_loss = (pred_a - target_a) ** 2 + (pred_b - target_b) ** 2
    sample_loss = sp.expand(sample_loss)
    print(f'  Per-sample loss(s | xa, xb, omega) =')
    print(f'    {sample_loss}')

    # Take expectation over xa, xb assumed independent N(0, 1):
    # E[xa^2] = E[xb^2] = 1, E[xa] = E[xb] = E[xa*xb] = 0.
    # So we replace xa^2 -> 1, xb^2 -> 1, xa*xb -> 0, xa -> 0, xb -> 0.
    expected = sample_loss
    expected = expected.subs(xa * xb, 0)
    expected = expected.subs(xa ** 2, 1)
    expected = expected.subs(xb ** 2, 1)
    expected = expected.subs(xa, 0).subs(xb, 0)
    expected = sp.simplify(expected)
    print(f'  Expected loss E[loss(s, omega)] under xa, xb ~ N(0, 1):')
    print(f'    {expected}')

    # Symbolic gradient w.r.t. s, set to zero, solve for s* in terms of omega:
    grad = sp.diff(expected, s)
    print(f'  d/ds E[loss] = {sp.simplify(grad)}')
    sols = sp.solve(grad, s)
    print(f'  Optimal s* (solving d/ds = 0): {sols}')


def main():
    print('Compiling symbolic loss and gradient...', end=' ')
    t0 = time.perf_counter()
    loss_fn, grad_fn = build_symbolic_loss_and_grad()
    print(f'done in {time.perf_counter() - t0:.3f}s')

    results = []
    for s_true, label in S_TRUE_VALUES:
        r = run_one(s_true, label, loss_fn, grad_fn)
        results.append(r)

    print_results(results)
    omega_probe()


if __name__ == '__main__':
    main()
