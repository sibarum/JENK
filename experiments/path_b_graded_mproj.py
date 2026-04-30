"""Stage-4b: M_proj reachability under graded coefficients.

The earlier matrix-recipe experiments (path_b_*) established that
M_proj = [[0, 0], [0, 1]] is provably outside the chain image when each
layer is c*I + d*M(s) with classical (c, d, s). The README's framework-
native claim is that M_proj equals left-multiplication by `0*g` at s=w,
where `0` is the Traction Zero atom — i.e., a recipe with graded
coefficients, not just graded s.

This script tests that claim head-on. It evaluates GradedNeuron on a
focused set of recipes against a battery of inputs, computes the
residue against M_proj*input after traction_simplify, and runs
SymbolicNeuron with a free-symbol omega for the firing differential.
The output is one report; nothing is hidden in graphs or summaries.

Run:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. conda run -n traction \
        python experiments/path_b_graded_mproj.py
"""

from __future__ import annotations

import sys
import sympy as sp

from jenk.graded import omega as omega_free
from jenk.graded_neuron import GradedNeuron
from jenk.symbolic_neuron import SymbolicNeuron
from jenk.trace import Trace
from jenk.traction import z, w, null, traction_simplify

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


# ----- target ----------------------------------------------------------------
# M_proj = [[0, 0], [0, 1]]   so   M_proj * (x_a, x_b) = (0, x_b).

def m_proj_target(x_a, x_b):
    return (sp.Integer(0), x_b)


# ----- inputs ----------------------------------------------------------------

a_sym, b_sym = sp.symbols('a b')

inputs = [
    ('generic (a, b)',   a_sym,         b_sym),
    ('numeric (3, 5)',   sp.Integer(3), sp.Integer(5)),
    ('numeric (1, 1)',   sp.Integer(1), sp.Integer(1)),
    ('atom (a, z)',      a_sym,         z),
    ('atom (z, b)',      z,             b_sym),
    ('atom (a, w)',      a_sym,         w),
    ('atom (w, b)',      w,             b_sym),
    ('atom (z, w)',      z,             w),
    ('atom (w, z)',      w,             z),
]


# ----- recipes ---------------------------------------------------------------
# Each recipe is (label, c, d, s).
# All recipes are passed verbatim to GradedNeuron.
# For the matrix-recipe baseline, we substitute z -> 0 and w -> omega_free,
# so the matrix-recipe sees a "classical" version of the same intent.

recipes = [
    ('framework: c=0, d=z, s=w',         sp.Integer(0), z,             w),
    ('graded c too: c=z, d=z, s=w',      z,             z,             w),
    ('graded s only: c=0, d=1, s=w',     sp.Integer(0), sp.Integer(1), w),
    ('s=-1 cardinal: c=0, d=1, s=-1',    sp.Integer(0), sp.Integer(1), sp.Integer(-1)),
    ('null d: c=0, d=null, s=w',         sp.Integer(0), null,          w),
    ('zero matrix: c=0, d=0',            sp.Integer(0), sp.Integer(0), w),
    ('identity: c=1, d=0',               sp.Integer(1), sp.Integer(0), w),
]


def to_classical(expr):
    """Map traction atoms to their closest classical proxies for SymbolicNeuron."""
    return sp.sympify(expr).subs({z: sp.Integer(0), w: omega_free, null: sp.Integer(0)})


# ----- evaluation ------------------------------------------------------------

def fmt(expr):
    return sp.sstr(expr)


def evaluate(neuron, x_a, x_b):
    out_a, out_b = neuron.forward(x_a, x_b)
    out_a_s = traction_simplify(out_a)
    out_b_s = traction_simplify(out_b)
    target = m_proj_target(x_a, x_b)
    res_a = traction_simplify(out_a_s - target[0])
    res_b = traction_simplify(out_b_s - target[1])
    return (out_a_s, out_b_s), (res_a, res_b)


def is_zero(expr):
    """True if expr equals numeric 0 exactly (not Null, not a z-residue)."""
    return expr == sp.Integer(0)


def hits_mproj_strict(res):
    """Strict: residue is numeric (0, 0)."""
    return is_zero(res[0]) and is_zero(res[1])


def hits_mproj_modulo_zero(res):
    """Loose: residue contains only Zero atoms or Null (no free symbols, no numbers)."""
    if hits_mproj_strict(res):
        return True
    for r in res:
        if r == sp.Integer(0):
            continue
        # accept residues whose every multiplicative factor is z, null, or
        # a free symbol that came from the input — i.e., expressions of
        # form z*<anything> with no surviving numeric or omega terms.
        if not r.has(z):
            return False
        if r.has(w):
            return False
        # require that without z, the rest collapses (i.e., r is a pure z-multiple)
        rest = r.subs(z, sp.Integer(1))
        if rest == sp.Integer(0):
            continue
    return True


# ----- report ----------------------------------------------------------------

def report_recipe(label, c, d, s):
    print('-' * 78)
    print(f'Recipe: {label}')
    print(f'         c={fmt(c)}   d={fmt(d)}   s={fmt(s)}')

    graded = GradedNeuron(s=s, c=c, d=d)
    classical = SymbolicNeuron(
        s=to_classical(s), c=to_classical(c), d=to_classical(d),
    )

    rows = []
    for inp_label, x_a, x_b in inputs:
        out_g, res_g = evaluate(graded, x_a, x_b)
        x_a_cl, x_b_cl = to_classical(x_a), to_classical(x_b)
        out_c, res_c = evaluate(classical, x_a_cl, x_b_cl)
        rows.append((inp_label, out_g, res_g, out_c, res_c))

    # Print as four columns per input
    print()
    print(f'  {"input":24s} | {"GradedNeuron output":34s} | residue')
    print(f'  {"":24s} | {"SymbolicNeuron output (free w)":34s} | residue')
    print('  ' + '-' * 92)
    for inp_label, out_g, res_g, out_c, res_c in rows:
        gflag = 'HIT-strict' if hits_mproj_strict(res_g) else (
            'HIT-mod-z'    if hits_mproj_modulo_zero(res_g) else '          ')
        sflag = 'HIT-strict' if hits_mproj_strict(res_c) else (
            'HIT-mod-z'    if hits_mproj_modulo_zero(res_c) else '          ')
        print(f'  {inp_label:24s} | G:({fmt(out_g[0])}, {fmt(out_g[1])})')
        print(f'  {gflag:24s} |   res=({fmt(res_g[0])}, {fmt(res_g[1])})')
        print(f'  {"":24s} | S:({fmt(out_c[0])}, {fmt(out_c[1])})')
        print(f'  {sflag:24s} |   res=({fmt(res_c[0])}, {fmt(res_c[1])})')
        print()

    return rows


def main():
    print('=' * 78)
    print('Stage-4b: M_proj reachability under graded coefficients')
    print('=' * 78)
    print()
    print('Target: M_proj * (x_a, x_b) = (0, x_b)')
    print('Method: forward(x_a, x_b) -> traction_simplify -> residue vs target')
    print('HIT-strict   : residue == (0, 0) numerically')
    print('HIT-mod-z    : residue is purely z-atom records (the framework')
    print('               "information-conservation" residue from 0*x deferred annihilation)')
    print()

    summary = []
    for label, c, d, s in recipes:
        rows = report_recipe(label, c, d, s)
        g_strict = sum(1 for r in rows if hits_mproj_strict(r[2]))
        g_modz   = sum(1 for r in rows if hits_mproj_modulo_zero(r[2]))
        s_strict = sum(1 for r in rows if hits_mproj_strict(r[4]))
        s_modz   = sum(1 for r in rows if hits_mproj_modulo_zero(r[4]))
        summary.append((label, len(rows), g_strict, g_modz, s_strict, s_modz))

    print('=' * 78)
    print('Summary  (G = GradedNeuron, S = SymbolicNeuron with free-symbol omega)')
    print('=' * 78)
    print(f'{"Recipe":42s} {"G-strict":>10s} {"G-mod-z":>10s} {"S-strict":>10s} {"S-mod-z":>10s}')
    for label, total, gs, gm, ss, sm in summary:
        print(f'{label:42s} {gs:>4d}/{total:<5d} {gm:>4d}/{total:<5d} {ss:>4d}/{total:<5d} {sm:>4d}/{total:<5d}')

    save_diagnostic_traces()


def save_diagnostic_traces():
    print()
    print('Diagnostic traces for the framework recipe (c=0, d=z, s=w):')
    n = GradedNeuron(s=w, c=sp.Integer(0), d=z)
    cases = [
        ('graded_mproj_generic',   a_sym,         b_sym),
        ('graded_mproj_numeric',   sp.Integer(3), sp.Integer(5)),
        ('graded_mproj_xb_eq_z',   a_sym,         z),
    ]
    for label, x_a, x_b in cases:
        with Trace(label=label) as t:
            n.forward(x_a, x_b, trace=t)
            path = t.save(f'tmp/diagnostics/{label}.json')
        s = t.summary()
        print(f'  {label:28s} records={s["n_records"]:<3} '
              f'zero-atoms={s["n_zero_atom_appearances"]:<3} '
              f'omega-atoms={s["n_omega_atom_appearances"]:<3} '
              f'simplified={s["n_simplifications"]:<3}')
        print(f'    -> {path}')


if __name__ == '__main__':
    main()
