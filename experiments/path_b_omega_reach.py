"""Path-B experiment #2: re-parametrizations to reach omega.

Three models:

  Model A - Direct real s.  Same as path_b_synthetic_recovery.
  Model B - s = sinh(theta).  Smooth 1-D bijection R -> R; gradient near
            large s is damped by 1/cosh(theta).  Omega still asymptotic
            (theta -> infinity) but convergence may be more uniform.
  Model C - Homogeneous (p, q) with M_homo(p, q) = [[0, -q], [q, p]].
            ORIGINAL M(s) recovered at q=1, p=s.  Omega = finite interior
            point (p=1, q=0); matrix becomes the projection [[0, 0], [0, 1]].
            Two parameters, but omega is genuinely reachable.

Probes:

  P1 - Recovery from finite s_true.  All three models should converge.
       Compare convergence rates as s_true grows.
  P2 - Limit / projection target.  Generate Y = [[0, 0], [0, 1]] X.
       Models A and B *cannot* express this matrix (M(s) always has
       (0, 1) = -1, never 0).  Only Model C can.  Confirm.
  P3 - Sanity:  Model C on finite-s data should still recover (s, 1).
       Verify it doesn't drift into the q=0 manifold.

Run from JENK/:
    PYTHONPATH=. conda run -n traction python tmp/path_b_omega_reach.py
"""

from __future__ import annotations

import time

import numpy as np
import sympy as sp


LR = 0.05
N_STEPS = 2000
SEED = 42
N_SAMPLES = 200


# ----------------------------------------------------------------- compile

def _compile_loss_grads(forward_a_expr, forward_b_expr, params, syms):
    """params: tuple of sympy symbols representing the trainable params.
    syms: tuple (xa, xb, ta, tb)."""
    xa, xb, ta, tb = syms
    loss = (forward_a_expr - ta) ** 2 + (forward_b_expr - tb) ** 2
    grads = [sp.diff(loss, p) for p in params]
    free = list(params) + list(syms)
    loss_fn = sp.lambdify(free, loss, 'numpy')
    grad_fns = [sp.lambdify(free, g, 'numpy') for g in grads]
    return loss_fn, grad_fns


def _sgd_step(param_values, loss_fn, grad_fns, X, Y, lr):
    xa, xb, ta, tb = X[:, 0], X[:, 1], Y[:, 0], Y[:, 1]
    args = list(param_values) + [xa, xb, ta, tb]
    loss = float(np.mean(loss_fn(*args)))
    grads = [float(np.mean(g(*args))) for g in grad_fns]
    new_params = [p - lr * g for p, g in zip(param_values, grads)]
    return new_params, loss, grads


def train(model, X, Y, init, n_steps=N_STEPS, lr=LR):
    """Generic training loop. Returns (final_params, loss_hist, params_hist)."""
    loss_fn, grad_fns = model
    params = [float(v) for v in init]
    loss_hist = []
    params_hist = [tuple(params)]
    for _ in range(n_steps):
        params, loss, _ = _sgd_step(params, loss_fn, grad_fns, X, Y, lr)
        loss_hist.append(loss)
        params_hist.append(tuple(params))
    return params, loss_hist, params_hist


# ---------------------------------------------------------------- models

def model_A_direct():
    """Model A: pred = M(s) X."""
    s = sp.Symbol('s', real=True)
    xa, xb, ta, tb = sp.symbols('xa xb ta tb', real=True)
    pred_a = -xb
    pred_b = s * xb + xa
    return _compile_loss_grads(pred_a, pred_b, (s,), (xa, xb, ta, tb))


def model_B_sinh():
    """Model B: pred = M(sinh(theta)) X.  s = sinh(theta)."""
    th = sp.Symbol('theta', real=True)
    xa, xb, ta, tb = sp.symbols('xa xb ta tb', real=True)
    s = sp.sinh(th)
    pred_a = -xb
    pred_b = s * xb + xa
    return _compile_loss_grads(pred_a, pred_b, (th,), (xa, xb, ta, tb))


def model_C_homogeneous():
    """Model C: M_homo(p, q) = [[0, -q], [q, p]];  pred = M_homo X."""
    p, q = sp.symbols('p q', real=True)
    xa, xb, ta, tb = sp.symbols('xa xb ta tb', real=True)
    pred_a = -q * xb
    pred_b = q * xa + p * xb
    return _compile_loss_grads(pred_a, pred_b, (p, q), (xa, xb, ta, tb))


# --------------------------------------------------------------- probes

def make_data_at_s(s_true: float, n: int, rng: np.random.Generator):
    X = rng.normal(size=(n, 2))
    M = np.array([[0.0, -1.0], [1.0, s_true]])
    Y = X @ M.T
    return X, Y


def make_data_at_projection(n: int, rng: np.random.Generator):
    """Generate Y = [[0, 0], [0, 1]] X — the omega-limit matrix."""
    X = rng.normal(size=(n, 2))
    M_proj = np.array([[0.0, 0.0], [0.0, 1.0]])
    Y = X @ M_proj.T
    return X, Y


def steps_until(loss_hist, threshold):
    for i, l in enumerate(loss_hist):
        if l < threshold:
            return i
    return None


# ---------------------------------------------------------------- P1: finite-s recovery

def probe_finite_s(model_A, model_B, model_C):
    print('#' * 92)
    print('# P1 - Recovery from finite s_true:  three models on Y = M(s_true) X')
    print('#' * 92)
    print()
    print(f"{'s_true':>9}  {'Model A: s_dir':>15}  {'A loss':>12}  "
          f"{'Model B: theta -> s':>26}  {'B loss':>12}  "
          f"{'Model C: (p, q)':>22}  {'C loss':>12}")
    print('-' * 121)
    for s_true in [1.0, 5.0, 50.0, 500.0]:
        rng = np.random.default_rng(SEED)
        X, Y = make_data_at_s(s_true, N_SAMPLES, rng)

        (sA,), lossA, _ = train(model_A, X, Y, init=(0.0,))
        (thB,), lossB, _ = train(model_B, X, Y, init=(0.0,))
        sB = float(np.sinh(thB))
        (pC, qC), lossC, _ = train(model_C, X, Y, init=(0.0, 0.5))

        print(f"{s_true:>9.1f}  {sA:>+15.6f}  {lossA[-1]:>12.3e}  "
              f"theta={thB:>+8.4f} -> s={sB:>+9.4f}  {lossB[-1]:>12.3e}  "
              f"({pC:>+7.4f}, {qC:>+7.4f})  {lossC[-1]:>12.3e}")
    print()


# ---------------------------------------------------------------- P2: omega-limit / projection

def probe_projection(model_A, model_B, model_C):
    print('#' * 92)
    print('# P2 - Omega-limit recovery:  Y = [[0, 0], [0, 1]] X (the q->0 projection)')
    print('#' * 92)
    print()
    print('Theory: M_proj has (0, 1) entry = 0, but every M(s) has (0, 1) = -1.')
    print('So Models A and B CANNOT express M_proj — must have residual loss.')
    print('Model C can hit it at (p=1, q=0).')
    print()
    rng = np.random.default_rng(SEED)
    X, Y = make_data_at_projection(N_SAMPLES, rng)

    print(f"{'model':>14}  {'final params':<32}  {'final loss':>12}  {'M_recovered':>30}")
    print('-' * 92)

    (sA,), lossA, _ = train(model_A, X, Y, init=(0.0,))
    M_A = np.array([[0.0, -1.0], [1.0, sA]])
    print(f"{'A: direct s':>14}  s = {sA:>+10.6f}{'':<19}  "
          f"{lossA[-1]:>12.3e}  [{M_A.ravel()}]")

    (thB,), lossB, _ = train(model_B, X, Y, init=(0.0,))
    sB = float(np.sinh(thB))
    M_B = np.array([[0.0, -1.0], [1.0, sB]])
    print(f"{'B: sinh(theta)':>14}  theta = {thB:>+10.4f}, s = {sB:>+10.4f}    "
          f"{lossB[-1]:>12.3e}  [{M_B.ravel()}]")

    (pC, qC), lossC, _ = train(model_C, X, Y, init=(0.0, 0.5), n_steps=N_STEPS)
    M_C = np.array([[0.0, -qC], [qC, pC]])
    print(f"{'C: homogeneous':>14}  (p, q) = ({pC:>+8.4f}, {qC:>+8.4f}){'':<8}  "
          f"{lossC[-1]:>12.3e}  [{M_C.ravel()}]")
    print()
    print('Watch:  does Model C settle on (p, q) close to (1, 0)?  That would be exact recovery.')
    print()


# ---------------------------------------------------------------- P3: Model C sanity at finite-s

def probe_modelC_sanity(model_C):
    print('#' * 92)
    print('# P3 - Model C sanity:  on finite-s data, does it recover (p=s, q=1)?')
    print('#' * 92)
    print()
    print(f"{'s_true':>9}  {'(p_recovered, q_recovered)':<38}  "
          f"{'effective s = p/q':>20}  {'final loss':>12}")
    print('-' * 92)
    for s_true in [-1.0, 0.5, 1.0, 2.5]:
        rng = np.random.default_rng(SEED)
        X, Y = make_data_at_s(s_true, N_SAMPLES, rng)
        (pC, qC), lossC, _ = train(model_C, X, Y, init=(0.0, 0.5))
        s_eff = pC / qC if abs(qC) > 1e-12 else float('inf')
        print(f"{s_true:>9.2f}  ({pC:>+10.6f}, {qC:>+10.6f}){'':<14}  "
              f"{s_eff:>+20.6f}  {lossC[-1]:>12.3e}")
    print()
    print('If recovery is correct, q should land near 1 and p near s_true (no')
    print('drift to a different equivalent point on the (p, q) line, since the')
    print('actual matrix is parameterized faithfully — there is no gauge freedom).')
    print()


# ---------------------------------------------------------------- P4: convergence-rate comparison

def probe_convergence_rates(model_A, model_B):
    print('#' * 92)
    print('# P4 - Convergence rate vs s_proxy:  does sinh re-param help for large s?')
    print('#' * 92)
    print()
    proxies = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    print(f"{'s_true':>8}  {'A: steps to <1e-6':>20}  {'B: steps to <1e-6':>20}  "
          f"{'A grad@init':>13}  {'B grad@init':>13}")
    print('-' * 92)
    for s_true in proxies:
        rng = np.random.default_rng(SEED)
        X, Y = make_data_at_s(s_true, N_SAMPLES, rng)

        (sA,), lossA, params_histA = train(model_A, X, Y, init=(0.0,))
        # find first step where |s - s_true| < 1e-6
        a_step = None
        for i, (sval,) in enumerate(params_histA):
            if abs(sval - s_true) < 1e-6:
                a_step = i
                break
        a_steps_str = str(a_step) if a_step is not None else f">{N_STEPS}"

        (thB,), lossB, params_histB = train(model_B, X, Y, init=(0.0,))
        b_step = None
        for i, (thval,) in enumerate(params_histB):
            if abs(np.sinh(thval) - s_true) < 1e-6:
                b_step = i
                break
        b_steps_str = str(b_step) if b_step is not None else f">{N_STEPS}"

        # gradient at init (s=0 / theta=0)
        loss_fn_A, grads_A = model_A
        loss_fn_B, grads_B = model_B
        xa, xb, ta, tb = X[:, 0], X[:, 1], Y[:, 0], Y[:, 1]
        gA0 = float(np.mean(grads_A[0](0.0, xa, xb, ta, tb)))
        gB0 = float(np.mean(grads_B[0](0.0, xa, xb, ta, tb)))

        print(f"{s_true:>8.0f}  {a_steps_str:>20}  {b_steps_str:>20}  "
              f"{gA0:>+13.3e}  {gB0:>+13.3e}")
    print()


# ---------------------------------------------------------------- main

def main():
    print('Compiling symbolic gradients for three models...', end=' ')
    t0 = time.perf_counter()
    A = model_A_direct()
    B = model_B_sinh()
    C = model_C_homogeneous()
    print(f'done in {time.perf_counter() - t0:.3f}s\n')

    probe_finite_s(A, B, C)
    probe_projection(A, B, C)
    probe_modelC_sanity(C)
    probe_convergence_rates(A, B)


if __name__ == '__main__':
    main()
