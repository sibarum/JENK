"""Diagnostic demo: GradedNeuron at s=w with framework-zero in the input.

Shows the trace mechanism in action — every intermediate captured, the
moments where w*z -> 1 fires, and the JSON dump for inspection.
"""

from __future__ import annotations

import sympy as sp

from jenk.graded_neuron import GradedNeuron, chain
from jenk.trace import Trace
from jenk.traction import z, w, null


def demo_single_neuron_at_omega():
    print('=' * 72)
    print('Demo 1: generator at s=w with x_b = z (framework zero)')
    print('=' * 72)
    print('Forward formula:')
    print('  out_a = c*x_a - d*x_b + bias_a')
    print('  out_b = d*x_a + (c + d*s)*x_b + bias_b')
    print()
    print('With c=0, d=1, s=w, x_a=a (symbol), x_b=z, biases=0:')
    print('  m4 = d*s = 1*w = w')
    print('  m6 = m5*x_b = w*z = 1   (FRAMEWORK IDENTITY FIRES)')
    print()

    n = GradedNeuron.generator(s=w)
    x_a = sp.Symbol('a')
    x_b = z

    with Trace(label='single_omega_z') as t:
        out_a, out_b = n.forward(x_a, x_b, trace=t)
        path = t.save('tmp/diagnostics/demo_single.json')

    print(f'out_a = {out_a}')
    print(f'out_b = {out_b}')
    print()
    print('Selected trace records (step / raw / simplified / flags):')
    for r in t.records:
        if r.flags['has_omega'] or r.flags['has_zero'] or 'm4' in r.step or 'm5' in r.step or 'm6' in r.step or 'output' in r.step:
            print(f'  {r.step:<35}  raw={r.raw["str"]:<20}  simp={r.simplified["str"]:<20}  flags={ {k:v for k,v in r.flags.items() if v} }')
    print()
    print(f'Summary: {t.summary()}')
    print(f'JSON saved to: {path}')


def demo_chain_at_two_cardinals():
    print()
    print('=' * 72)
    print('Demo 2: chain of generator(s=0) then generator(s=w)')
    print('=' * 72)

    n1 = GradedNeuron.generator(s=sp.Integer(0))
    n2 = GradedNeuron.generator(s=w)
    composed = chain(n1, n2)

    a, b = sp.Symbol('a'), sp.Symbol('b')

    with Trace(label='chain_0_then_w') as t:
        out_a, out_b = composed(a, b, trace=t)
        path = t.save('tmp/diagnostics/demo_chain.json')

    print(f'out_a = {out_a}')
    print(f'out_b = {out_b}')
    print(f'Summary: {t.summary()}')
    print(f'JSON saved to: {path}')


if __name__ == '__main__':
    demo_single_neuron_at_omega()
    demo_chain_at_two_cardinals()
