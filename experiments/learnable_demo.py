"""Stage-5 demo: symbolic gradient descent on a learnable `s` parameter.

Goal: train one LearnableNeuron's `s` to converge to a known target cardinal
(0 = hyperbolic, 1 = parabolic, -1 = elliptic), with the entire forward
pass and gradient expression remaining symbolic. Every training step is
recorded into a Trace including the symbolic gradient itself.

Why this is the right shape for Stage 5
---------------------------------------

The M_proj experiments established that real-valued `s` reaches three of
four cardinals smoothly; ω is a discrete substitution outside the chain
image. So a learnable real-valued `s` is the natural first training
target — it exercises gradient flow without engaging the spike question.

Run:

    PYTHONIOENCODING=utf-8 PYTHONPATH=. conda run -n traction \
        python experiments/learnable_demo.py
"""

from __future__ import annotations

import sys
import sympy as sp

from jenk.learnable_neuron import (
    LearnableParameter,
    LearnableNeuron,
    SymbolicOptimizer,
    squared_error_loss,
)
from jenk.trace import Trace

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def make_pairs_for_target(s_target: float) -> list[tuple[tuple, tuple]]:
    """Generate input/output pairs consistent with c=0, d=1, s=s_target.

    Forward formula at c=0, d=1: out_a = -x_b, out_b = x_a + s_target * x_b.
    """
    inputs = [(1, 1), (1, -1), (2, 3), (-1, 2)]
    return [
        ((x_a, x_b), (-x_b, x_a + s_target * x_b))
        for x_a, x_b in inputs
    ]


def run_convergence(label: str, s_target: float, s_initial: float, *,
                    learning_rate: float, n_steps: int) -> dict:
    """Train one neuron's s to converge to s_target. Trace every step."""
    print(f'\n{"=" * 72}')
    print(f'{label}: s_initial={s_initial} -> s_target={s_target}')
    print(f'{"=" * 72}')

    s = LearnableParameter.named('s', s_initial)
    n = LearnableNeuron(s=s, c=0, d=1)

    pairs = make_pairs_for_target(s_target)
    loss_expr = squared_error_loss(n, pairs)
    opt = SymbolicOptimizer(n, learning_rate=learning_rate)

    print(f'Loss expression (symbolic, in s):')
    print(f'  {sp.simplify(loss_expr)}')
    grad_expr = opt.gradient_expr(loss_expr, s)
    print(f'Gradient ∂loss/∂s (symbolic):')
    print(f'  {sp.simplify(grad_expr)}')
    print()

    with Trace(label=f'learnable_demo_{label}') as t:
        # Capture initial conditions
        t.record('initial.s_value', sp.Float(s.value),
                 note=f'starting position; target = {s_target}')
        t.record('symbolic.loss_expr', loss_expr,
                 note='symbolic loss expression in s — fixed across training')
        t.record('symbolic.grad_expr', grad_expr,
                 note='symbolic ∂loss/∂s — fixed across training')

        # Print a header every 10 steps for inspection
        print(f'  {"step":>5}  {"s":>14}  {"loss":>14}  {"grad":>14}')
        for step in range(n_steps):
            loss_before = opt.loss_value(loss_expr)
            grad_now = opt.gradient_value(loss_expr, s)
            t.record(f'step{step:03d}.s_value', sp.Float(s.value))
            t.record(f'step{step:03d}.loss', sp.Float(loss_before))
            t.record(f'step{step:03d}.grad_value', sp.Float(grad_now))

            if step < 20 or step % 20 == 0:
                print(f'  {step:>5d}  {s.value:>14.8f}  '
                      f'{loss_before:>14.8e}  {grad_now:>14.8e}')

            opt.step(loss_expr)

        final_loss = opt.loss_value(loss_expr)
        final_s = s.value
        t.record('final.s_value', sp.Float(final_s),
                 note=f'final position; target was {s_target}')
        t.record('final.loss', sp.Float(final_loss))
        t.record('final.s_error', sp.Float(abs(final_s - s_target)))

        path = t.save(f'tmp/diagnostics/learnable_demo_{label}.json')

    print()
    print(f'Final s = {final_s:.10f} (target {s_target}, error {abs(final_s - s_target):.2e})')
    print(f'Final loss = {final_loss:.4e}')
    print(f'Trace: {path}')
    print(f'Trace records: {len(t.records)}')

    return {
        'label': label,
        's_target': s_target,
        's_final': final_s,
        'final_loss': final_loss,
        'trace_path': path,
    }


def main() -> None:
    print('=' * 72)
    print('Stage-5 demo: symbolic gradient descent on learnable s')
    print('=' * 72)
    print()
    print('Each run trains one neuron with c=0, d=1 and a learnable real s,')
    print('converging to one of the three real cardinals (0, 1, -1).')
    print('The symbolic loss and gradient expressions are unchanging across')
    print('training; only their numeric evaluation moves with s.')

    runs = []
    runs.append(run_convergence(
        'hyperbolic', s_target=0.0, s_initial=1.5,
        learning_rate=0.02, n_steps=200,
    ))
    runs.append(run_convergence(
        'parabolic', s_target=1.0, s_initial=-0.5,
        learning_rate=0.02, n_steps=200,
    ))
    runs.append(run_convergence(
        'elliptic', s_target=-1.0, s_initial=0.5,
        learning_rate=0.02, n_steps=200,
    ))

    print()
    print('=' * 72)
    print('Summary')
    print('=' * 72)
    print(f'{"label":<12} {"target":>8} {"final":>14} {"error":>12} {"loss":>14}')
    for r in runs:
        print(f'{r["label"]:<12} {r["s_target"]:>8.2f} {r["s_final"]:>14.10f} '
              f'{abs(r["s_final"] - r["s_target"]):>12.2e} {r["final_loss"]:>14.2e}')

    print()
    print('Trace JSONs include:')
    print('  - symbolic.loss_expr  : the unchanging loss as a sympy expression in s')
    print('  - symbolic.grad_expr  : the unchanging ∂loss/∂s symbolic gradient')
    print('  - step{N}.{s_value, loss, grad_value} for each step')
    print('  - final.{s_value, loss, s_error}')
    print()
    print('Inspect with: cat tmp/diagnostics/learnable_demo_<label>.json')


if __name__ == '__main__':
    main()
