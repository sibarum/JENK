"""closure_explorer.py - Jenk closure investigation, graded carrier.

Carrier and wrap rules now live in jenk.graded; this script only
holds the candidate framework and the two probes (non-cardinal s,
cardinal products).

Two questions on the table:
  (1) Can a closed-form g(s) -> (a(s), b(s)) hit  g(s)^2 = 0^s = (1, s)
      at the four cardinal s in {0, 1, w, -1}?
  (2) What does the projective case (s = -1, target g^2 = w = (1, -1))
      require?
"""

from dataclasses import dataclass

import sympy as sp

from jenk.graded import CARDINALS, Graded, omega

# `s` is the candidate-parameter symbol. Distinct from the Chebyshev `s`
# in jenk.ring (which is a *real* trace parameter, not a sympy symbol).
s = sp.Symbol('s', real=True)


# --- Candidate framework ---------------------------------------------------

@dataclass
class Candidate:
    name: str
    a_expr: sp.Expr   # coefficient component, function of `s`
    b_expr: sp.Expr   # grade component,       function of `s`

    def at(self, s_value) -> Graded:
        return Graded(self.a_expr.subs(s, s_value),
                      self.b_expr.subs(s, s_value))

    def squared_at(self, s_value) -> Graded:
        return self.at(s_value).squared().reduced()


def _check(label: str, ok: bool) -> str:
    return "OK" if ok else "BREAK"


def _fmt_target(target: Graded) -> str:
    target_red = target.reduced()
    same = (sp.simplify(target.a - target_red.a) == 0
            and sp.simplify(target.b - target_red.b) == 0)
    return f"{target}" if same else f"{target}  ->  {target_red}"


def check_cardinals(c: Candidate) -> None:
    print(f"=== Cardinals: {c.name} ===")
    print(f"    g(s) = ({c.a_expr},  {c.b_expr})")
    for s_val, target, mode, unit in CARDINALS:
        got = c.squared_at(s_val)
        target_red = target.reduced()
        diff_a = sp.simplify(got.a - target_red.a)
        diff_b = sp.simplify(got.b - target_red.b)
        ok_a = (diff_a == 0)
        ok_b = (diff_b == 0)
        if ok_a and ok_b:
            marker = "OK  "
        elif ok_a or ok_b:
            marker = "PART"
        else:
            marker = "MISS"
        comp_status = f"a:{'OK' if ok_a else '..'}  b:{'OK' if ok_b else '..'}"
        print(f"    [{marker}]  s={str(s_val):>5}  ({mode:>11}, {unit})  {comp_status}")
        print(f"           got    = {got}")
        print(f"           target = {_fmt_target(target)}")


def check_closure(c: Candidate) -> None:
    """Symbolic closure check:  g(s1) * g(s2)  ?=  g(s1 + s2)."""
    s1, s2 = sp.symbols('s1 s2', real=True)
    a_lhs = sp.simplify(c.a_expr.subs(s, s1) * c.a_expr.subs(s, s2))
    b_lhs = sp.simplify(c.b_expr.subs(s, s1) + c.b_expr.subs(s, s2))
    a_rhs = sp.simplify(c.a_expr.subs(s, s1 + s2))
    b_rhs = sp.simplify(c.b_expr.subs(s, s1 + s2))
    diff_a = sp.simplify(a_lhs - a_rhs)
    diff_b = sp.simplify(b_lhs - b_rhs)
    print(f"=== Closure (g(s1)*g(s2) ?= g(s1+s2)): {c.name} ===")
    print(f"    coefficient:  lhs - rhs = {diff_a}    [{_check('a', diff_a == 0)}]")
    print(f"    grade:        lhs - rhs = {diff_b}    [{_check('b', diff_b == 0)}]")


def probe_non_cardinal_s(c: Candidate, s_values) -> None:
    """Show g(s) and g(s)^2 (reduced) at non-cardinal s values.
    Probes whether the carrier is finer than the four cardinals - i.e.
    whether it gives well-defined values 'between' them.
    """
    print(f"=== Non-cardinal s probe: {c.name} ===")
    print(f"    g(s) = ({c.a_expr},  {c.b_expr})")
    print(f"    {'s':>10}    {'g(s)':<24}    {'g(s)^2 (reduced)':<24}    note")
    for s_val, note in s_values:
        g = Graded(c.a_expr.subs(s, s_val), c.b_expr.subs(s, s_val)).simplified()
        g_sq = c.squared_at(s_val)
        print(f"    {str(s_val):>10}    {str(g):<24}    {str(g_sq):<24}    {note}")


def probe_cardinal_products(c: Candidate) -> None:
    """Compute pairwise products of the four cardinal units (j, e, n, k)
    in the graded carrier, and report each in reduced form. Shakes out:
      - whether unit*unit recovers the right delta
      - whether 0 * omega = 1 falls out structurally (e.g. e * k)
      - whether off-cardinal products land on non-cardinal grades
    """
    print(f"=== Cardinal unit products ({c.name}) ===")
    units = []
    for s_val, _target, mode, sym in CARDINALS:
        u = Graded(c.a_expr.subs(s, s_val), c.b_expr.subs(s, s_val)).simplified()
        units.append((sym, mode, s_val, u))
    print("    Units (g(s) at the cardinal s):")
    for sym, mode, s_val, u in units:
        print(f"      {sym}  ({mode:>11})  =  g(s={s_val})  =  {u}")
    print("    Pairwise products X * Y (reduced):")
    syms = [u[0] for u in units]
    header = "         " + "  ".join(f"{sy:^22}" for sy in syms)
    print(header)
    for sym1, _m1, _s1, u1 in units:
        cells = []
        for _sym2, _m2, _s2, u2 in units:
            prod = (u1 * u2).reduced()
            cells.append(f"{str(prod):^22}")
        print(f"    {sym1}    " + "  ".join(cells))


# --- Candidates ------------------------------------------------------------

# (a) The previous purely complex rotation, lifted into the graded carrier
# by setting b = 0 identically. Same content as before; included for
# comparison.
ROTATION = Candidate(
    name='rotation: (exp(i*pi*s/2), 0)',
    a_expr=sp.exp(sp.I * sp.pi * s / 2),
    b_expr=sp.Integer(0),
)

# (b) Pure grading: g(s) = (1, s/2). Then g(s)^2 = (1, s) = 0^s by
# definition. Hits j, e, k exactly. Misses n only because reducing
# (1, w) to (-1, 0) requires applying the README identity 0^w = -1,
# which is *not* a structural rule of the carrier (it's an extra rewrite).
GRADING = Candidate(
    name='grading: (1, s/2)',
    a_expr=sp.Integer(1),
    b_expr=s / 2,
)

# (c) Hybrid: keep the grading, fold a complex rotation into the
# coefficient with frequency tied to omega so it winds exactly once
# from s=0 to s=w. Probe for whether this picks up the elliptic wrap
# without an explicit rewrite rule.
HYBRID = Candidate(
    name='hybrid: (exp(i*pi*s/(2*omega)), s/2)',
    a_expr=sp.exp(sp.I * sp.pi * s / (2 * omega)),
    b_expr=s / 2,
)

CANDIDATES = [ROTATION, GRADING, HYBRID]


NON_CARDINAL_S = [
    (sp.Rational(1, 2),    "between j and e ; expect g^2 = 0^(1/2) = e itself"),
    (sp.Rational(-1, 2),   "between j and k ; expect g^2 = 0^(-1/2) = k itself"),
    (sp.Integer(2),        "deeper zero (0^2)"),
    (sp.Integer(-2),       "deeper omega (omega^2) ; sign-flip should reduce to (1, 2)"),
    (sp.Integer(3),        "third-level zero"),
    (omega / 2,            "elliptic root 0^(omega/2) = eta itself"),
    (2 * omega,            "full cycle ; elliptic wrap should reduce g^2 to 1"),
    ((omega + 1) / 2,      "non-closed expression (per README): 0^(w/2)*0^(1/2)"),
]


def main():
    for c in CANDIDATES:
        check_cardinals(c)
        check_closure(c)
        print()

    # Pure-grading is the cleanest candidate; probe with it.
    probe_non_cardinal_s(GRADING, NON_CARDINAL_S)
    print()
    probe_cardinal_products(GRADING)


if __name__ == '__main__':
    main()
