"""Chebyshev ring  Q[s][g] / (g^2 - s*g + 1)  with sympy-valued s.

Lifted variant of jenk.ring: the trace parameter `s` and the coefficients
`a`, `b` are sympy expressions, which may contain `omega` from
jenk.graded. Ring arithmetic uses sympy throughout, so

  - at any rational `s`, `a`, `b`, the lifted ring agrees numerically
    with jenk.ring once converted back to floats;
  - at `s = omega` (or any other sympy expression), the lifted ring
    stays symbolic and is closed under +, -, *, conjugate, norm, and
    inverse-where-norm-is-non-zero.

Defining identity is unchanged from jenk.ring:

    (a1 + b1*g)(a2 + b2*g)
        = (a1*a2 - b1*b2) + (a1*b2 + b1*a2 + b1*b2*s) * g.

Two operands must share the same `s` (under sp.simplify(s1-s2)==0) to
compose; mixed-s arithmetic raises ValueError.

This layer intentionally does NOT bake the README's graded wrap rules
(0^omega = -1, omega = -0, etc.) into ring operations. Those identities
are still being explored; canonicalization here is plain sp.simplify.
Callers can apply Graded.reduced() or sp.subs externally if they want
to project results onto the graded carrier.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from jenk import ring as float_ring


def _sympify(value) -> sp.Expr:
    """Coerce a numeric or symbolic input to a sympy expression."""
    return sp.sympify(value)


def _equal(left: sp.Expr, right: sp.Expr) -> bool:
    """Decide left == right under sympy simplify; falls back to False on indeterminate."""
    diff = sp.simplify(left - right)
    return diff == 0


def _require_same_s(left: 'LiftedRingElement', right: 'LiftedRingElement', op: str) -> None:
    if not _equal(left.s, right.s):
        raise ValueError(
            f"Lifted ring {op} requires matching s; got {left.s} and {right.s}"
        )


@dataclass(frozen=True)
class LiftedRingElement:
    """Element  a + b*g  in  (sympy-expr-in-omega)[s][g] / (g^2 - s*g + 1).

    `a`, `b`, and `s` are stored as sympy expressions. Inputs are sympified
    in __post_init__, so passing Python ints/floats or sp.Rational works
    interchangeably.
    """
    a: sp.Expr
    b: sp.Expr
    s: sp.Expr

    def __post_init__(self) -> None:
        # frozen dataclass: bypass via object.__setattr__ to coerce inputs.
        object.__setattr__(self, 'a', _sympify(self.a))
        object.__setattr__(self, 'b', _sympify(self.b))
        object.__setattr__(self, 's', _sympify(self.s))

    # --- Constructors ------------------------------------------------------

    @staticmethod
    def identity(s) -> 'LiftedRingElement':
        return LiftedRingElement(sp.Integer(1), sp.Integer(0), s)

    @staticmethod
    def generator(s) -> 'LiftedRingElement':
        return LiftedRingElement(sp.Integer(0), sp.Integer(1), s)

    # --- Additive structure -----------------------------------------------

    def add(self, other: 'LiftedRingElement') -> 'LiftedRingElement':
        _require_same_s(self, other, 'addition')
        return LiftedRingElement(self.a + other.a, self.b + other.b, self.s)

    def sub(self, other: 'LiftedRingElement') -> 'LiftedRingElement':
        _require_same_s(self, other, 'subtraction')
        return LiftedRingElement(self.a - other.a, self.b - other.b, self.s)

    def negate(self) -> 'LiftedRingElement':
        return LiftedRingElement(-self.a, -self.b, self.s)

    def scale(self, k) -> 'LiftedRingElement':
        ks = _sympify(k)
        return LiftedRingElement(self.a * ks, self.b * ks, self.s)

    # --- Multiplicative structure -----------------------------------------

    def mult(self, other: 'LiftedRingElement') -> 'LiftedRingElement':
        """Ring product via g^2 = s*g - 1."""
        _require_same_s(self, other, 'multiplication')
        a1, b1, s = self.a, self.b, self.s
        a2, b2 = other.a, other.b
        new_a = a1 * a2 - b1 * b2
        new_b = a1 * b2 + b1 * a2 + b1 * b2 * s
        return LiftedRingElement(new_a, new_b, s)

    def conjugate(self) -> 'LiftedRingElement':
        """conj(a + b*g) = (a + b*s) - b*g."""
        return LiftedRingElement(self.a + self.b * self.s, -self.b, self.s)

    def norm(self) -> sp.Expr:
        """Norm  N(a + b*g) = a^2 + a*b*s + b^2  as a sympy expression.

        Equals the scalar component of x*conj(x); the g-component vanishes
        identically. Returns a sympy expression rather than a float, so
        norms can stay symbolic in s, omega, etc.
        """
        return self.a * self.a + self.a * self.b * self.s + self.b * self.b

    def inverse(self) -> 'LiftedRingElement':
        """Multiplicative inverse  x^{-1} = conj(x) / N(x).

        Raises ZeroDivisionError when sp.simplify(N(x)) == 0. Symbolic
        norms that simplify can resolve to zero (e.g., on the light cone
        s=2 with a=b=1), and we treat those as singular.
        """
        n = sp.simplify(self.norm())
        if n == 0:
            raise ZeroDivisionError(
                f"LiftedRingElement has zero norm "
                f"(s={self.s}, a={self.a}, b={self.b})"
            )
        return self.conjugate().scale(sp.Integer(1) / n)

    # --- Classification ---------------------------------------------------

    def signature_region(self) -> str:
        """'elliptic'/'parabolic'/'hyperbolic' when s is a real number, else 'symbolic'.

        For symbolic s (e.g., s=omega, s=t with t a free symbol) the
        signature is undefined without further substitution; callers can
        pin down s via .subs and re-check.
        """
        s_simp = sp.simplify(self.s)
        if s_simp.is_number and s_simp.is_real:
            abs_s = abs(float(s_simp))
            if abs_s < 2.0 - float_ring._PARABOLIC_EPS:
                return 'elliptic'
            if abs_s > 2.0 + float_ring._PARABOLIC_EPS:
                return 'hyperbolic'
            return 'parabolic'
        return 'symbolic'

    # --- Canonicalization / substitution ---------------------------------

    def simplified(self) -> 'LiftedRingElement':
        """Apply sp.simplify to a, b, and s component-wise."""
        return LiftedRingElement(sp.simplify(self.a),
                                 sp.simplify(self.b),
                                 sp.simplify(self.s))

    def subs(self, *args, **kwargs) -> 'LiftedRingElement':
        """Apply sympy substitution to a, b, and s. Same signature as sp.Expr.subs."""
        return LiftedRingElement(self.a.subs(*args, **kwargs),
                                 self.b.subs(*args, **kwargs),
                                 self.s.subs(*args, **kwargs))

    # --- Operator overloads (sugar over the named methods) ---------------

    def __add__(self, other):
        if isinstance(other, LiftedRingElement):
            return self.add(other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, LiftedRingElement):
            return self.sub(other)
        return NotImplemented

    def __neg__(self):
        return self.negate()

    def __mul__(self, other):
        if isinstance(other, LiftedRingElement):
            return self.mult(other)
        try:
            ks = _sympify(other)
        except (sp.SympifyError, TypeError):
            return NotImplemented
        return self.scale(ks)

    def __rmul__(self, other):
        try:
            ks = _sympify(other)
        except (sp.SympifyError, TypeError):
            return NotImplemented
        return self.scale(ks)

    def __repr__(self) -> str:
        return f"LiftedRingElement({self.a} + ({self.b})*g | s={self.s})"
