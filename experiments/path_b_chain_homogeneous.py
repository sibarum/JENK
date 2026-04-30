"""Re-test of Test 4 (chain expressiveness) using the homogeneous (p, q)
parametrization that won the omega-reach experiment.

Each neuron is now M_homo(p, q) = [[0, -q], [q, p]] (2 params per neuron).
A 2-neuron chain has 4 params total. The chain product is

  M_homo(p2, q2) M_homo(p1, q1) = [[ -q1*q2,  -q2*p1     ],
                                    [ p2*q1,  p1*p2 - q1*q2 ]]

Probes:

  Q1 - Same target as the original Test 4: Y = M(s_total=1.5) X.
       Algebraically the chain still can't fit it exactly:
         (0, 0) = -q1*q2 = 0  forces q1 = 0 OR q2 = 0;
         q1 = 0  =>  (1, 0) = 0 != 1;   q2 = 0  =>  (0, 1) = 0 != -1.
       What does the optimizer settle on, and how does it compare to
       the generator-only chain's settle?

  Q2 - Target the chain CAN reach: pick concrete (p1*, q1*, p2*, q2*),
       compute the matrix product, generate data from it.  Train the
       chain and verify recovery.  (Also: is there a gauge symmetry?
       (p1, q1) and (p2, q2) appear paired in products; rescaling
       one vs. the other might leave the product invariant.  Probe.)

  Q3 - Initialize at the omega-limit (q close to 0) and see whether
       optimization stays well-behaved or collapses.

Run from JENK/:
    PYTHONPATH=. conda run -n traction python tmp/path_b_chain_homogeneous.py
"""

from __future__ import annotations

import time

import numpy as np
import sympy as sp


LR = 0.05
N_STEPS = 2000
SEED = 42
N_SAMPLES = 200


def compile_chain():
    """Build the symbolic 2-neuron M_homo chain and lambdify loss/grads."""
    p1, q1, p2, q2 = sp.symbols('p1 q1 p2 q2', real=True)
    xa, xb, ta, tb = sp.symbols('xa xb ta tb', real=True)

    # Layer 1
    mid_a = -q1 * xb
    mid_b = q1 * xa + p1 * xb

    # Layer 2
    out_a = -q2 * mid_b
    out_b = q2 * mid_a + p2 * mid_b

    loss = (out_a - ta) ** 2 + (out_b - tb) ** 2
    grads = [sp.diff(loss, p) for p in (p1, q1, p2, q2)]
    free = (p1, q1, p2, q2, xa, xb, ta, tb)
    return (
        sp.lambdify(free, loss, 'numpy'),
        [sp.lambdify(free, g, 'numpy') for g in grads],
        (out_a, out_b),
    )


def train_chain(loss_fn, grad_fns, X, Y, init, n_steps=N_STEPS, lr=LR):
    p = list(map(float, init))  # p1, q1, p2, q2
    xa, xb, ta, tb = X[:, 0], X[:, 1], Y[:, 0], Y[:, 1]
    losses = []
    for _ in range(n_steps):
        l = float(np.mean(loss_fn(*p, xa, xb, ta, tb)))
        gs = [float(np.mean(g(*p, xa, xb, ta, tb))) for g in grad_fns]
        losses.append(l)
        p = [pi - lr * gi for pi, gi in zip(p, gs)]
    return tuple(p), losses


def chain_product_matrix(p1, q1, p2, q2):
    M1 = np.array([[0.0, -q1], [q1, p1]])
    M2 = np.array([[0.0, -q2], [q2, p2]])
    return M2 @ M1


def make_data_from_matrix(M, n, rng):
    X = rng.normal(size=(n, 2))
    Y = X @ M.T
    return X, Y


# ============================================================ Q1

def Q1_same_target_as_original_test4(loss_fn, grad_fns):
    print('#' * 92)
    print('# Q1 - Same target as original Test 4:  Y = M(s_total=1.5) X')
    print('#' * 92)
    print()

    s_total = 1.5
    M_target = np.array([[0.0, -1.0], [1.0, s_total]])
    rng = np.random.default_rng(SEED)
    X, Y = make_data_from_matrix(M_target, N_SAMPLES, rng)

    inits = [
        (0.0, 0.5, 0.0, 0.5),  # both q=0.5
        (1.0, 1.0, 1.0, 1.0),
        (0.5, 1.0, 0.5, 1.0),
        (-1.0, 0.5, 2.0, 0.5),
        (3.0, 1.0, -1.0, 1.0),
        (1.5, 1.0, 0.0, 1.0),  # one neuron at the s_total
        (0.0, 0.1, 0.0, 0.1),  # near omega
    ]
    print(f"{'init (p1, q1, p2, q2)':<32}  "
          f"{'final (p1, q1, p2, q2)':<36}  "
          f"{'final loss':>12}  {'fit M_chain[0,0]':>17}")
    print('-' * 100)
    for init in inits:
        params, losses = train_chain(loss_fn, grad_fns, X, Y, init)
        M_chain = chain_product_matrix(*params)
        init_str = '(' + ', '.join(f'{v:+.2f}' for v in init) + ')'
        final_str = '(' + ', '.join(f'{v:+.4f}' for v in params) + ')'
        print(f"{init_str:<32}  {final_str:<36}  "
              f"{losses[-1]:>12.3e}  {M_chain[0, 0]:>+17.4f}")
    print()
    print('Theory: the chain image at (0, 0) = -q1*q2 = 0 forces q1=0 or q2=0,')
    print('then (1, 0) or (0, 1) cannot match  =>  irreducible residual.')
    print('Optimizer should settle on the closest-fit and report nonzero loss.')
    print()


# ============================================================ Q2

def Q2_reachable_target(loss_fn, grad_fns):
    print('#' * 92)
    print('# Q2 - Target the M_homo chain CAN reach:  generate from chain itself')
    print('#' * 92)
    print()

    # Pick a chain-reachable target by composing two M_homo with chosen params.
    p1_t, q1_t, p2_t, q2_t = 0.7, 1.3, 1.5, 0.4
    M_target = chain_product_matrix(p1_t, q1_t, p2_t, q2_t)
    print(f'  ground truth: (p1, q1, p2, q2) = ({p1_t}, {q1_t}, {p2_t}, {q2_t})')
    print(f'  resulting M_target =')
    print(f'    {M_target.tolist()}')
    print()

    rng = np.random.default_rng(SEED)
    X, Y = make_data_from_matrix(M_target, N_SAMPLES, rng)

    inits = [
        (0.0, 0.5, 0.0, 0.5),
        (1.0, 1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0, 1.0),
        (0.5, 0.5, 1.0, 0.5),
        (p1_t, q1_t, p2_t, q2_t),  # init at the truth
    ]
    print(f"{'init':<32}  {'final (p1, q1, p2, q2)':<36}  "
          f"{'final loss':>12}  {'M_chain matches target':>23}")
    print('-' * 110)
    for init in inits:
        params, losses = train_chain(loss_fn, grad_fns, X, Y, init)
        M_chain = chain_product_matrix(*params)
        match = float(np.linalg.norm(M_chain - M_target, 'fro'))
        init_str = '(' + ', '.join(f'{v:+.2f}' for v in init) + ')'
        final_str = '(' + ', '.join(f'{v:+.4f}' for v in params) + ')'
        print(f"{init_str:<32}  {final_str:<36}  "
              f"{losses[-1]:>12.3e}  ||M-M*||_F = {match:.3e}")
    print()
    print('The product matrix has a 1-parameter gauge symmetry:')
    print('  M_homo(p2, q2) M_homo(p1, q1)  is  invariant under (p1, q1) -> (lambda*p1, lambda*q1)')
    print('  if we simultaneously rescale (p2, q2) -> (p2/lambda, q2/lambda).')
    print('Verify: -q1 q2 -> -lambda q1 (q2/lambda) = -q1 q2  (unchanged).')
    print('So different inits should converge to *equivalent* (p1, q1, p2, q2) on the same orbit.')
    print()


# ============================================================ Q3

def Q3_init_near_omega(loss_fn, grad_fns):
    print('#' * 92)
    print('# Q3 - Init near the omega-limit (one or both q small):  optimization stable?')
    print('#' * 92)
    print()
    rng = np.random.default_rng(SEED)
    s_total = 1.5
    M_target = np.array([[0.0, -1.0], [1.0, s_total]])
    X, Y = make_data_from_matrix(M_target, N_SAMPLES, rng)

    inits = [
        (1.0, 1e-6, 1.0, 1.0),    # q1 near 0
        (1.0, 1.0, 1.0, 1e-6),    # q2 near 0
        (1.0, 1e-6, 1.0, 1e-6),   # both near 0
        (1.0, 0.0, 1.0, 0.0),     # both EXACTLY 0
    ]
    print(f"{'init':<36}  {'final (p1, q1, p2, q2)':<40}  {'final loss':>12}")
    print('-' * 100)
    for init in inits:
        try:
            params, losses = train_chain(loss_fn, grad_fns, X, Y, init)
            init_str = '(' + ', '.join(f'{v:+.2e}' for v in init) + ')'
            final_str = '(' + ', '.join(f'{v:+.4f}' for v in params) + ')'
            print(f"{init_str:<36}  {final_str:<40}  {losses[-1]:>12.3e}")
        except Exception as exc:
            print(f"  init {init}  FAILED: {exc}")
    print()
    print('Hopes: gradients near q=0 are still finite because the loss is polynomial')
    print('in (p, q).  The optimizer should escape from q=0 if the target needs q != 0.')
    print()


def main():
    print('Compiling symbolic 2-neuron M_homo chain...', end=' ')
    t0 = time.perf_counter()
    loss_fn, grad_fns, _ = compile_chain()
    print(f'done in {time.perf_counter() - t0:.3f}s\n')

    Q1_same_target_as_original_test4(loss_fn, grad_fns)
    Q2_reachable_target(loss_fn, grad_fns)
    Q3_init_near_omega(loss_fn, grad_fns)


if __name__ == '__main__':
    main()
