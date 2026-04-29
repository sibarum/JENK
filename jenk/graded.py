"""Graded carrier for the four-mode (a, b) representation.

A Graded element  (a, b)  carries the semantics
        (a, b)  =  a * 0^b  =  a * w^(-b)
where w (omega) is the README's projective unit (1/0). Multiplication is
component-wise:
        (a1, b1) * (a2, b2)  =  (a1 * a2,  b1 + b2).
Squaring follows:
        (a, b)^2  =  (a^2, 2*b).

This preserves the 0/w tower as distinct grades:
        (1, 0) = 1,   (1, 1) = 0,   (1, -1) = w,   (1, 2) = 0^2,   (1, -2) = w^2.

Two README identities are applied as rewrites in `Graded.reduced()`:
    (i)  Elliptic wrap    (0^w = -1):     (a, b + k*w)  ->  ((-1)^k * a, b)  for integer k
    (ii) Sign-flip wrap   (w = -0):       (a, k)        ~   ((-1)^k * a, -k)  for integer k

The four cardinals from the README, expressed as targets for g(s)^2
where g(s) = 0^(s/2):
        s=0    -> (1,  0)    (1,            hyperbolic j,  d=+1)
        s=1    -> (1,  1)    (0,            parabolic  e,  d= 0)
        s=w    -> (-1, 0)    (-1,           elliptic   n,  d=-1; via 0^w = -1 wrap)
        s=-1   -> (1, -1)    (w,            projective k,  d= w)

omega is exposed as a free sympy symbol with a positivity assumption so
sympy can simplify w/w -> 1 where it would otherwise stall. It carries
no numerical value - any numerical proxy is the caller's choice.
"""

from dataclasses import dataclass

import sympy as sp

# w (omega): projective unit, free symbol. Positivity lets sympy reduce
# w/w -> 1 in expressions where it would otherwise leave the ratio.
omega = sp.Symbol('omega', positive=True)


@dataclass(frozen=True)
class Graded:
    """Pair (a, b) representing  a * 0^b  =  a * w^(-b)."""
    a: sp.Expr
    b: sp.Expr

    def __mul__(self, other: 'Graded') -> 'Graded':
        return Graded(self.a * other.a, self.b + other.b)

    def squared(self) -> 'Graded':
        return Graded(self.a ** 2, 2 * self.b)

    def simplified(self) -> 'Graded':
        return Graded(sp.simplify(self.a), sp.simplify(self.b))

    def reduced(self) -> 'Graded':
        """Apply README rewrites in addition to sympy simplify.

        Currently applies:
          (i)  Elliptic wrap (0^w = -1):
                  (a, b + k*w)  ->  ((-1)^k * a, b)   for integer k.
          (ii) Sign-flip wrap (w = -0):
                  (a, k)        ~   ((-1)^k * a, -k)  for integer k.
                  Canonicalizes the grade to b >= 0 when b is a concrete integer.

        Note: (0 * w = 1) and (w^(-1) = 0) are already enforced by the
        carrier's structural multiplication
                (1, 1) * (1, -1) = (1, 0)
        without an extra rewrite.

        TODO: more identities from the README that could become reductions
        on this carrier - revisit if we keep this representation:
          - 0 + w = 0           -> additive cross-grade collapse
          - x - x = erasure     -> the Roman-zero collapse
          - 0^(w/2) = +/- i     -> half-integer w (the elliptic unit)
          - w^w = -1            -> non-linear-in-w wraps  (b = c * w^k)
          - 0^2, w^2 vs 0, w    -> unit-circle vs deeper-grade
                                   equivalences if the README intends any
        Each is a separate design call and may interact with the existing
        rules; take them one at a time.
        """
        g = self.simplified()
        a, b = g.a, g.b
        # (i) Elliptic wrap: extract integer multiple of w from b.
        k_omega = b.coeff(omega)
        if k_omega != 0 and k_omega.is_integer:
            k_int = int(k_omega)
            sgn = sp.Integer(-1 if k_int % 2 else 1)
            a = sp.simplify(a * sgn)
            b = sp.simplify(b - k_omega * omega)
        # (ii) Sign-flip wrap: when b is a concrete negative integer,
        # send (a, b) -> ((-1)^b * a, -b) so the canonical form has b >= 0.
        if b.is_integer and b < 0:
            k_int = int(b)
            sgn = sp.Integer(-1 if k_int % 2 else 1)
            a = sp.simplify(a * sgn)
            b = sp.simplify(-b)
        return Graded(a, b)

    def __repr__(self) -> str:
        return f"({self.a}, {self.b})"


# README cardinals: target value of g(s)^2 = 0^s for the four distinguished s.
# Each row is (s_value, target_g_squared, mode_name, unit_symbol).
# These are the canonical README forms (not yet reduced); .reduced() on
# the projective target will canonicalize (1, -1) -> (-1, 1) via sign-flip.
CARDINALS = [
    (sp.Integer(0),    Graded(sp.Integer(1),  sp.Integer(0)),         'hyperbolic', 'j'),
    (sp.Integer(1),    Graded(sp.Integer(1),  sp.Integer(1)),         'parabolic',  'e'),
    (omega,            Graded(sp.Integer(-1), sp.Integer(0)),         'elliptic',   'n'),
    (sp.Integer(-1),   Graded(sp.Integer(1),  sp.Integer(-1)),        'projective', 'k'),
]
