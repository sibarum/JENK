"""Stage-5b: inductive-bias test — LearnableNeuron vs FreeLinearLayer.

Question: does the framework's structural prior (1 learnable parameter,
fixed `c=0, d=1`) give a measurable advantage over an unconstrained
free 2x2 linear layer (4 learnable parameters) when fitting Möbius
data with noise and limited samples?

Hypothesis: yes. The 1-param model can't overfit noise into matrix
entries that don't belong to the ring image. The 4-param model can,
and should generalize worse on a held-out clean test set when
training data is small or noisy.

Method:
  - Pick a target s_target in (-2, 2) (inside the elliptic / hyperbolic
    transition region — somewhere both modes have to "sound right").
  - Generate train pairs from M(c=0, d=1, s_target) with Gaussian
    noise added to the outputs. Generate a clean held-out test set.
  - Train each layer for a fixed number of steps with squared-error loss.
  - Report: training loss, test loss, recovered s (or matrix entries),
    error in s, error in matrix Frobenius norm.
  - Repeat across data sizes, noise levels, multiple seeds. Median over
    seeds for each cell.

Run:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. conda run -n traction \
        python experiments/path_b_inductive_bias.py
"""

from __future__ import annotations

import sys
from statistics import median

import numpy as np
import sympy as sp

from jenk.free_linear import FreeLinearLayer
from jenk.learnable_neuron import (
    LearnableNeuron,
    LearnableParameter,
    SymbolicOptimizer,
    squared_error_loss,
)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def generate_pairs(s_target: float, n: int, noise_std: float, rng: np.random.Generator):
    """Generate n input/output pairs from M(c=0, d=1, s_target) + Gaussian noise."""
    pairs = []
    for _ in range(n):
        x_a = float(rng.uniform(-1.0, 1.0))
        x_b = float(rng.uniform(-1.0, 1.0))
        clean_a = -x_b
        clean_b = x_a + s_target * x_b
        out_a = clean_a + float(rng.normal(0.0, noise_std))
        out_b = clean_b + float(rng.normal(0.0, noise_std))
        pairs.append(((x_a, x_b), (out_a, out_b)))
    return pairs


def train_layer(layer, pairs, lr: float, n_steps: int) -> float:
    """Train a layer on pairs. Returns the final training loss."""
    loss_expr = squared_error_loss(layer, pairs)
    opt = SymbolicOptimizer(layer, learning_rate=lr)
    for _ in range(n_steps):
        opt.step(loss_expr)
    return opt.loss_value(loss_expr)


def evaluate_loss(layer, pairs) -> float:
    """Compute squared-error loss on pairs at current parameter values."""
    loss_expr = squared_error_loss(layer, pairs)
    opt = SymbolicOptimizer(layer)
    return opt.loss_value(loss_expr)


def matrix_error(free: FreeLinearLayer, s_target: float) -> float:
    """Frobenius distance from free's matrix to the target M(s_target)."""
    m11, m12, m21, m22 = free.matrix_values()
    target = np.array([[0.0, -1.0], [1.0, s_target]])
    actual = np.array([[m11, m12], [m21, m22]])
    return float(np.linalg.norm(target - actual))


def run_trial(s_target: float, n_train: int, noise_std: float, *,
              seed: int, n_steps: int = 300, lr: float = 0.05,
              n_test: int = 200) -> dict:
    """One trial: generate data, train both layers, return metrics."""
    rng = np.random.default_rng(seed)
    train_pairs = generate_pairs(s_target, n_train, noise_std, rng)
    test_pairs = generate_pairs(s_target, n_test, 0.0, rng)  # clean held-out

    # ---- LearnableNeuron ----
    s_param = LearnableParameter.named('s', 0.0)
    learnable = LearnableNeuron(s=s_param, c=0, d=1)
    learnable_train_loss = train_layer(learnable, train_pairs, lr=lr, n_steps=n_steps)
    learnable_test_loss = evaluate_loss(learnable, test_pairs) / n_test
    s_recovered = s_param.value
    s_error = abs(s_recovered - s_target)

    # ---- FreeLinearLayer ----
    free = FreeLinearLayer.zero_init()
    free_train_loss = train_layer(free, train_pairs, lr=lr, n_steps=n_steps)
    free_test_loss = evaluate_loss(free, test_pairs) / n_test
    matrix_err = matrix_error(free, s_target)

    return {
        'learnable_train_loss': learnable_train_loss / n_train,
        'learnable_test_loss': learnable_test_loss,
        's_recovered': s_recovered,
        's_error': s_error,
        'free_train_loss': free_train_loss / n_train,
        'free_test_loss': free_test_loss,
        'matrix_error': matrix_err,
        'free_matrix': free.matrix_values(),
    }


def run_cell(s_target: float, n_train: int, noise_std: float, n_seeds: int) -> dict:
    """Run multiple seeds, return median of each metric."""
    trials = [
        run_trial(s_target, n_train, noise_std, seed=42 + i)
        for i in range(n_seeds)
    ]
    return {
        'n_train': n_train,
        'noise_std': noise_std,
        'learnable_train_loss': median(t['learnable_train_loss'] for t in trials),
        'learnable_test_loss': median(t['learnable_test_loss'] for t in trials),
        's_error': median(t['s_error'] for t in trials),
        'free_train_loss': median(t['free_train_loss'] for t in trials),
        'free_test_loss': median(t['free_test_loss'] for t in trials),
        'matrix_error': median(t['matrix_error'] for t in trials),
        'n_seeds': n_seeds,
    }


def main() -> None:
    s_target = 0.7  # off-cardinal so both have to actually learn
    n_seeds = 5
    data_sizes = [3, 5, 10, 30]
    noise_levels = [0.0, 0.05, 0.2, 0.5]

    print('=' * 88)
    print('Stage-5b: inductive-bias test — LearnableNeuron vs FreeLinearLayer')
    print('=' * 88)
    print(f'Target:   s_target = {s_target}')
    print(f'Setup:    LearnableNeuron has 1 free param (s); FreeLinear has 4 (full 2x2)')
    print(f'Trial:    {n_seeds} seeds per (n_train, noise) cell, median reported')
    print(f'Training: 300 steps at lr=0.05, squared-error loss')
    print(f'Test:     200 clean held-out pairs, mean per-pair squared error')
    print()

    print(f'{"n_train":>8} {"noise":>7}  '
          f'{"L_train":>10} {"L_test":>10} {"s_err":>9}  '
          f'{"F_train":>10} {"F_test":>10} {"M_err":>9}  '
          f'{"L/F test":>10}')
    print('-' * 100)

    rows = []
    for n_train in data_sizes:
        for noise in noise_levels:
            r = run_cell(s_target, n_train, noise, n_seeds)
            rows.append(r)
            l_test = r['learnable_test_loss']
            f_test = r['free_test_loss']
            ratio = l_test / f_test if f_test > 1e-15 else float('inf')
            print(
                f'{n_train:>8d} {noise:>7.2f}  '
                f'{r["learnable_train_loss"]:>10.2e} {l_test:>10.2e} '
                f'{r["s_error"]:>9.2e}  '
                f'{r["free_train_loss"]:>10.2e} {f_test:>10.2e} '
                f'{r["matrix_error"]:>9.2e}  '
                f'{ratio:>10.2e}'
            )

    print()
    print('Legend:')
    print('  L_train, L_test, s_err   : LearnableNeuron train loss, test loss, |recovered s - s_target|')
    print('  F_train, F_test, M_err   : FreeLinear  train loss, test loss, ||recovered M - target M||_F')
    print('  L/F test                 : ratio of LearnableNeuron test loss to FreeLinear test loss')
    print('                              (< 1.0 = LearnableNeuron generalizes better)')
    print()
    print('Reading the table:')
    print('  - Both columns L_train and F_train should be similar at the noise floor.')
    print('  - L_test and F_test diverge most at small n_train + high noise — this is')
    print('    where the 1-param structural prior pays off.')
    print('  - s_err and M_err measure recovery of the underlying structure.')


if __name__ == '__main__':
    main()
