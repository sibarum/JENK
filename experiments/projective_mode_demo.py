"""Live demo: what the symbolic neuron actually outputs at each cardinal s.

Run from JENK/:
    conda run -n traction python tmp/projective_mode_demo.py
"""
import sympy as sp

from jenk.graded import omega
from jenk.symbolic_neuron import SymbolicNeuron, chain


x_a, x_b = sp.symbols('x_a x_b', real=True)


def _print_cardinal_table():
    print('=== Single generator-weight neuron at each cardinal s ===')
    print('  weight = g (c=0, d=1), input = (x_a, x_b)')
    print()
    cases = [
        (sp.Integer(0),  'hyperbolic mode j'),
        (sp.Integer(1),  'parabolic  mode e'),
        (omega,          'elliptic   mode n'),
        (sp.Integer(-1), 'projective mode k'),
    ]
    for s_val, name in cases:
        n = SymbolicNeuron.generator(s_val)
        out = n.forward(x_a, x_b)
        g2 = n.carrier_g_squared()
        print(f'  s = {str(s_val):>5}  ({name})')
        print(f'    matrix M(s)        = {n.matrix().tolist()}')
        print(f'    forward output     = ({out[0]}, {out[1]})')
        print(f'    carrier g(s)^2     = {g2}')
        print()


def _print_composition():
    print('=== Composition: projective(-1) -> elliptic(omega) -> hyperbolic(0) ===')
    composed = chain(
        SymbolicNeuron.generator(sp.Integer(-1)),
        SymbolicNeuron.generator(omega),
        SymbolicNeuron.generator(sp.Integer(0)),
    )
    out = composed(x_a, x_b)
    print(f'  output[0] = {sp.expand(out[0])}')
    print(f'  output[1] = {sp.expand(out[1])}')
    print()


def _print_omega_substitution_probes():
    print('=== Probes: substitute omega with various proxies ===')
    n = SymbolicNeuron.generator(omega)
    out = n.forward(sp.Integer(1), sp.Integer(1))
    print(f'  Input (1, 1) at s=omega -> output = {out}')
    print(f'    sub omega -> 0 :     {(out[0].subs(omega, 0), out[1].subs(omega, 0))}')
    print(f'    sub omega -> 1 :     {(out[0].subs(omega, 1), out[1].subs(omega, 1))}')
    print(f'    sub omega -> -1:     {(out[0].subs(omega, -1), out[1].subs(omega, -1))}')
    print(f'    sub omega -> oo:     {(out[0].subs(omega, sp.oo), out[1].subs(omega, sp.oo))}')
    print()


def main():
    _print_cardinal_table()
    _print_composition()
    _print_omega_substitution_probes()


if __name__ == '__main__':
    main()
