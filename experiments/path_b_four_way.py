"""Four-way benchmark: MLP vs Free Linear vs Complex-valued vs (c, d, s)-chain.

The point: a credible reviewer will ask why a vanilla MLP can't do whatever the
ring-chain does. This run answers that on tasks spanning the four cardinal
regimes plus the omega-limit projection.

Architectures (all 2-in/2-out, all trained with full-batch SGD, lr=0.05, 1000 steps):

  FreeLinear  W (2x2)   .....................   4 params,  no nonlinearity
  Complex1L   z -> w*z (rotation+scale only) ...  2 params,  no nonlinearity
  MLP-tanh    2 -> 4 -> 2 with tanh activations  18 params,  nonlinear
  RingChain   chain of 2 (c, d, s) neurons      6 params,  no nonlinearity

The first three are baselines; only RingChain is the model under test.

Tasks:  Y = M_target X  for the following targets

  s = 0           rotation 90 deg          (hyperbolic-mode cardinal j)
  s = 1           rotation 60 deg          (parabolic-mode cardinal e)
  s = -1          rotation 120 deg         (projective-mode cardinal k)
  s = 3           hyperbolic boost         (|s| > 2 region)
  M_proj          [[0, 0], [0, 1]]          (omega-limit projection)

Predictions before running:

  - Complex1L can express only [[a, -b], [b, a]] rotations+scalings. It will
    fit s=0, s=1, s=-1 (all rotations) exactly, but FAIL on s=3 (hyperbolic
    boost; needs symmetric eigenvalues) and on M_proj (singular projection,
    not a rotation+scale).  Closed-form best-fit losses:  s=3 -> 4.5,
    M_proj -> 0.5.

  - FreeLinear can express any 2x2 and should fit everything to machine eps.

  - MLP-tanh has tanh squashing; with enough capacity it can approximate
    linear functions but not exactly. Expected: very low loss but not eps.

  - RingChain (c, d, s) per neuron, 6 params:  predicted to fit everything
    given the analysis from the chain-homogeneous re-test (one neuron acts
    as scalar-identity, the other carries the trace).

Run from JENK/:
    PYTHONPATH=. conda run -n traction python tmp/path_b_four_way.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sp


SEED = 42
LR = 0.05
N_STEPS = 1000
N_TEST = 2000


# --------------------------------------------------------------- data

@dataclass
class Task:
    name: str
    M: np.ndarray  # the 2x2 target

    def generate(self, n: int, sigma: float, rng: np.random.Generator):
        X = rng.normal(size=(n, 2))
        Y = X @ self.M.T
        if sigma > 0:
            Y = Y + sigma * rng.normal(size=Y.shape)
        return X, Y


def M_of_s(s):
    return np.array([[0.0, -1.0], [1.0, float(s)]])


TASKS = [
    Task('s=0  (rot 90)',   M_of_s(0)),
    Task('s=1  (rot 60)',   M_of_s(1)),
    Task('s=-1 (rot 120)',  M_of_s(-1)),
    Task('s=3  (hyper)',    M_of_s(3)),
    Task('M_proj  (omega)', np.array([[0.0, 0.0], [0.0, 1.0]])),
]


# --------------------------------------------------------------- architectures
# Each architecture is a function  train(X, Y) -> (n_params, predict_fn, train_loss_history)
# predict_fn(X_test) -> Y_test_pred.

def train_free_linear(X, Y, n_steps=N_STEPS, lr=LR, rng=None):
    rng = rng or np.random.default_rng(SEED)
    W = rng.normal(size=(2, 2)) * 0.1
    n = X.shape[0]
    losses = []
    for _ in range(n_steps):
        err = X @ W.T - Y
        losses.append(float(np.mean(err ** 2)))
        W = W - lr * (2.0 / n) * (err.T @ X)

    def predict(X_test):
        return X_test @ W.T
    return 4, predict, losses


def train_complex_1layer(X, Y, n_steps=N_STEPS, lr=LR, rng=None):
    """Complex-valued 1-layer:  z -> w * z  with z = xa + i*xb.
    Implements  W = [[a, -b], [b, a]]  with 2 real params."""
    rng = rng or np.random.default_rng(SEED)
    a = float(rng.normal() * 0.1)
    b = float(rng.normal() * 0.1)
    n = X.shape[0]
    losses = []
    for _ in range(n_steps):
        # forward
        pred_a = a * X[:, 0] - b * X[:, 1]
        pred_b = b * X[:, 0] + a * X[:, 1]
        err_a = pred_a - Y[:, 0]
        err_b = pred_b - Y[:, 1]
        loss = float(np.mean(err_a ** 2 + err_b ** 2))
        losses.append(loss)
        # gradients
        # d/da:  pred_a -> +xa,  pred_b -> +xb
        # d/db:  pred_a -> -xb,  pred_b -> +xa
        da = (2.0 / n) * np.sum(err_a * X[:, 0] + err_b * X[:, 1])
        db = (2.0 / n) * np.sum(err_a * (-X[:, 1]) + err_b * X[:, 0])
        a = a - lr * da
        b = b - lr * db

    def predict(X_test):
        out_a = a * X_test[:, 0] - b * X_test[:, 1]
        out_b = b * X_test[:, 0] + a * X_test[:, 1]
        return np.column_stack([out_a, out_b])
    return 2, predict, losses


def train_mlp(X, Y, hidden=4, n_steps=N_STEPS, lr=LR, rng=None):
    """MLP  2 -> hidden -> 2  with tanh."""
    rng = rng or np.random.default_rng(SEED)
    scale = 1.0 / np.sqrt(2)
    W1 = rng.normal(size=(hidden, 2)) * scale
    b1 = np.zeros(hidden)
    W2 = rng.normal(size=(2, hidden)) * (1.0 / np.sqrt(hidden))
    b2 = np.zeros(2)
    n = X.shape[0]
    losses = []
    for _ in range(n_steps):
        z1 = X @ W1.T + b1                 # (n, H)
        h = np.tanh(z1)
        y = h @ W2.T + b2                  # (n, 2)
        err = y - Y                        # (n, 2)
        loss = float(np.mean(err ** 2))
        losses.append(loss)
        dY = (2.0 / n) * err
        dW2 = dY.T @ h
        db2 = dY.sum(axis=0)
        dh = dY @ W2
        dz1 = dh * (1.0 - h ** 2)
        dW1 = dz1.T @ X
        db1 = dz1.sum(axis=0)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    def predict(X_test):
        z1 = X_test @ W1.T + b1
        h = np.tanh(z1)
        return h @ W2.T + b2
    n_params = hidden * 2 + hidden + 2 * hidden + 2  # W1+b1 + W2+b2
    return n_params, predict, losses


# Ring chain: 2 neurons each with (c, d, s).  Build symbolic forward+grad once.

def _build_ring_chain():
    c1, d1, s1, c2, d2, s2 = sp.symbols('c1 d1 s1 c2 d2 s2', real=True)
    xa, xb, ta, tb = sp.symbols('xa xb ta tb', real=True)

    # layer 1:  out = M(c, d, s) [xa, xb]^T
    a1 = c1 * xa - d1 * xb
    b1 = d1 * xa + (c1 + d1 * s1) * xb
    # layer 2
    a2 = c2 * a1 - d2 * b1
    b2 = d2 * a1 + (c2 + d2 * s2) * b1

    loss = (a2 - ta) ** 2 + (b2 - tb) ** 2
    params = (c1, d1, s1, c2, d2, s2)
    free = params + (xa, xb, ta, tb)
    grads = [sp.diff(loss, p) for p in params]
    loss_fn = sp.lambdify(free, loss, 'numpy')
    grad_fns = [sp.lambdify(free, g, 'numpy') for g in grads]
    forward_a_fn = sp.lambdify(params + (xa, xb), a2, 'numpy')
    forward_b_fn = sp.lambdify(params + (xa, xb), b2, 'numpy')
    return loss_fn, grad_fns, forward_a_fn, forward_b_fn


_RING_CHAIN_FNS = None


def _ring_chain_fns():
    global _RING_CHAIN_FNS
    if _RING_CHAIN_FNS is None:
        _RING_CHAIN_FNS = _build_ring_chain()
    return _RING_CHAIN_FNS


def train_ring_chain(X, Y, n_steps=N_STEPS, lr=LR, rng=None):
    rng = rng or np.random.default_rng(SEED)
    loss_fn, grad_fns, fwd_a, fwd_b = _ring_chain_fns()
    # Init: small random for c, d; zero for s.  Avoid exact c=d=0 which gives
    # zero gradient (degenerate).
    c1, d1, s1, c2, d2, s2 = (
        float(rng.normal() * 0.3),
        float(rng.normal() * 0.3 + 0.5),  # nudge d away from 0
        float(rng.normal() * 0.1),
        float(rng.normal() * 0.3),
        float(rng.normal() * 0.3 + 0.5),
        float(rng.normal() * 0.1),
    )
    xa, xb, ta, tb = X[:, 0], X[:, 1], Y[:, 0], Y[:, 1]
    losses = []
    for _ in range(n_steps):
        l = float(np.mean(loss_fn(c1, d1, s1, c2, d2, s2, xa, xb, ta, tb)))
        losses.append(l)
        gs = [float(np.mean(g(c1, d1, s1, c2, d2, s2, xa, xb, ta, tb)))
              for g in grad_fns]
        c1 -= lr * gs[0]; d1 -= lr * gs[1]; s1 -= lr * gs[2]
        c2 -= lr * gs[3]; d2 -= lr * gs[4]; s2 -= lr * gs[5]

    def predict(X_test):
        out_a = fwd_a(c1, d1, s1, c2, d2, s2, X_test[:, 0], X_test[:, 1])
        out_b = fwd_b(c1, d1, s1, c2, d2, s2, X_test[:, 0], X_test[:, 1])
        return np.column_stack([out_a, out_b])
    return 6, predict, losses


# --------------------------------------------------------------- runner

ARCHS = [
    ('FreeLinear  ', train_free_linear),
    ('Complex1L   ', train_complex_1layer),
    ('MLP-tanh    ', train_mlp),
    ('RingChain   ', train_ring_chain),
]


def run_one_setting(task: Task, sigma: float, n_train: int, n_test=N_TEST):
    """Run all four architectures on (task, sigma, n_train); return dict of test losses."""
    rng_data = np.random.default_rng(SEED)
    X_train, Y_train = task.generate(n_train, sigma, rng_data)
    rng_test = np.random.default_rng(SEED + 7)
    X_test, Y_test = task.generate(n_test, sigma=0.0, rng=rng_test)

    results = {}
    for arch_name, train_fn in ARCHS:
        rng_init = np.random.default_rng(SEED + hash(arch_name) % 1000)
        n_p, predict, train_losses = train_fn(X_train, Y_train, rng=rng_init)
        Y_pred = predict(X_test)
        test_loss = float(np.mean((Y_pred - Y_test) ** 2))
        results[arch_name] = {
            'n_params': n_p,
            'final_train_loss': train_losses[-1],
            'test_loss': test_loss,
        }
    return results


def fmt_loss(x):
    return f'{x:.3e}'


def header_line(items):
    return '  '.join(items)


def report(settings_results):
    """settings_results: list of (label, results_dict)."""
    arch_names = [a for a, _ in ARCHS]

    # Header
    print()
    print('Tests:  Y = M X    (clean test loss on 2000 noiseless held-out samples)')
    print()
    cells = [f"{'setting':<48}"] + [f"{a.strip():>13}" for a in arch_names]
    print(header_line(cells))
    print(' ' * 4 + 'param counts: ' + ', '.join(
        f"{a.strip()}={settings_results[0][1][a]['n_params']}" for a in arch_names
    ))
    print('-' * (48 + 15 * len(arch_names)))
    for label, results in settings_results:
        row = [f'{label:<48}']
        for a in arch_names:
            row.append(f"{results[a]['test_loss']:>13.3e}")
        print(header_line(row))
    print()


def main():
    print('Compiling ring-chain symbolic gradients...', end=' ')
    t0 = time.perf_counter()
    _ring_chain_fns()
    print(f'done in {time.perf_counter() - t0:.3f}s\n')

    # ---- Pass 1:  noiseless, abundant data (200 samples) -------------------
    print('#' * 92)
    print('# PASS 1 - Noiseless, n=200:  who can fit each target exactly?')
    print('#' * 92)
    settings = []
    for task in TASKS:
        results = run_one_setting(task, sigma=0.0, n_train=200)
        settings.append((task.name, results))
    report(settings)

    # ---- Pass 2:  noisy + limited (50 samples, sigma=0.5) ------------------
    print('#' * 92)
    print('# PASS 2 - Noisy (sigma=0.5), limited (n=20):  inductive-bias regime')
    print('#' * 92)
    settings = []
    for task in TASKS:
        results = run_one_setting(task, sigma=0.5, n_train=20)
        settings.append((task.name, results))
    report(settings)

    # ---- Pass 3:  small data, no noise (n=4) -- minimum information --------
    print('#' * 92)
    print('# PASS 3 - Tiny data (n=4), no noise:  who generalizes from 4 examples?')
    print('#' * 92)
    settings = []
    for task in TASKS:
        results = run_one_setting(task, sigma=0.0, n_train=4)
        settings.append((task.name, results))
    report(settings)

    # ---- Closed-form verification for Complex1L on impossible targets ------
    print('#' * 92)
    print('# Sanity:  closed-form best-fit for Complex1L on impossible targets')
    print('#' * 92)
    print()
    print('Complex1L can express only [[a, -b], [b, a]].  Best L2-fit to M is')
    print('a* = (M[0,0] + M[1,1]) / 2,  b* = (M[1,0] - M[0,1]) / 2,')
    print('with residual = sum of (M - W*) entries squared.')
    print()
    for task in TASKS:
        M = task.M
        a_star = (M[0, 0] + M[1, 1]) / 2.0
        b_star = (M[1, 0] - M[0, 1]) / 2.0
        W_star = np.array([[a_star, -b_star], [b_star, a_star]])
        residual = float(np.sum((M - W_star) ** 2))
        print(f'  {task.name:<22}  a*={a_star:+.4f}  b*={b_star:+.4f}  '
              f'expected min train loss  -> {residual:.3e}')


if __name__ == '__main__':
    main()
