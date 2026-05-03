"""Chebyshev ring  Q[t][g] / (g² − t·g + 1)  with Traction-aware symbolic algebra.

This is a fresh, self-contained Chebyshev ring built on the framework's
sympy atoms from `jenk.traction`. Every coefficient is a sympy expression
that may contain Zero, Omega, or Null; every arithmetic result is passed
through `traction_simplify`, so framework identities (0·ω = 1, 0^ω = −1,
ω⁻¹ = 0, etc.) fire automatically when they apply.

Convention reminder
-------------------
The trace parameter is named `t` here, deliberately. In the rest of the
codebase / framework, `s` denotes the four-mode parameter
(g(s) = 0^(s/2)) and `t` denotes the Chebyshev trace (g + g⁻¹ = t,
g² = t·g − 1). The two are different rings; this module is the
Chebyshev one. Don't pass mode-parameter values where a trace is
expected — they're not interchangeable.

Design
------
A `ChebyshevRing` instance is a ring "context" carrying a single trace
parameter `t` (a sympy symbol or any sympy expression, including a
Traction atom or a numeric constant). Every `RingElement` belongs to a
specific `ChebyshevRing`; arithmetic between elements requires they share
the same ring (same `t`), checked structurally.

An element is a pair `(a, b)` representing `a + b·g`, with `g² = t·g − 1`.
Both `a` and `b` are sympy expressions that may include `t`, may include
Traction atoms, and may include any other sympy symbols.

Specialization
--------------
Use `element.subs(t, value)` to specialize the trace to a specific value:
a number, another sympy symbol, or a Traction atom. The resulting element
lives in a new ring; the original is unchanged. For example:

    R = ChebyshevRing()  # t is a free sympy symbol
    g = R.generator()
    g_at_omega = g.subs(R.t, Omega())  # g² = ω·g − 1, with ω the Traction atom

Norm and inverse
----------------
The ring norm  N(a + b·g) = a² + a·b·t + b²  is the scalar component of
x · conj(x). Inverses exist when the norm is non-zero (sympy-checkable);
on the Chebyshev "light cone" (|t| = 2 with a²+ab·t+b² = 0) the inverse
raises ZeroDivisionError.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from old.jenk.traction import traction_simplify


def _simplify(expr: sp.Expr) -> sp.Expr:
    """Apply traction_simplify, falling back to sp.simplify if it can't handle the expr."""
    try:
        return traction_simplify(expr)
    except Exception:
        return sp.simplify(expr)


def _equal(left: sp.Expr, right: sp.Expr) -> bool:
    """Structural equality up to expansion + traction_simplify on the difference.

    Polynomial differences in free symbols (e.g. t) need sp.expand to canonicalize;
    Traction-atom identities need traction_simplify. Apply both to the difference
    so 48·t² + 100·t + 20 == 20 + 100·t + 48·t² and (0·ω) == 1 both hold here.
    """
    diff = sp.expand(left - right)
    return _simplify(diff) == 0


@dataclass(frozen=True)
class ChebyshevRing:
    """A Chebyshev ring instance with trace parameter `t`.

    `t` may be a free sympy symbol (the default), a numeric constant, a
    Traction atom (Zero, Omega, Null), or any sympy expression.

    Use the constructors `zero()`, `one()`, `generator()`, or `element(a, b)`
    to build elements. Arithmetic operators are defined on RingElement.
    """
    t: sp.Expr = field(default_factory=lambda: sp.Symbol('t'))

    def __post_init__(self) -> None:
        object.__setattr__(self, 't', sp.sympify(self.t))

    def element(self, a, b) -> 'RingElement':
        """Construct  a + b·g  in this ring."""
        return RingElement(self, sp.sympify(a), sp.sympify(b))

    def zero(self) -> 'RingElement':
        return self.element(0, 0)

    def one(self) -> 'RingElement':
        return self.element(1, 0)

    def generator(self) -> 'RingElement':
        return self.element(0, 1)

    def from_scalar(self, k) -> 'RingElement':
        """Embed a sympy scalar  k  as  k + 0·g."""
        return self.element(k, 0)

    def __repr__(self) -> str:
        return f'ChebyshevRing(t={self.t})'


def _require_same_ring(left: 'RingElement', right: 'RingElement', op: str) -> None:
    if left.ring is right.ring:
        return
    if _equal(left.ring.t, right.ring.t):
        return
    raise ValueError(
        f'Chebyshev ring {op} requires matching trace; '
        f'got t={left.ring.t} and t={right.ring.t}'
    )


@dataclass(frozen=True)
class RingElement:
    """An element  a + b·g  of a ChebyshevRing.

    Both `a` and `b` are sympy expressions; they may contain the ring's
    trace `t`, Traction atoms, or any other free symbols.

    Arithmetic results are passed through `traction_simplify`, so framework
    identities collapse where they apply.
    """
    ring: ChebyshevRing
    a: sp.Expr
    b: sp.Expr

    def __post_init__(self) -> None:
        object.__setattr__(self, 'a', sp.sympify(self.a))
        object.__setattr__(self, 'b', sp.sympify(self.b))

    @property
    def t(self) -> sp.Expr:
        return self.ring.t

    # --- Additive structure ---------------------------------------------

    def add(self, other: 'RingElement') -> 'RingElement':
        _require_same_ring(self, other, 'addition')
        return self.ring.element(_simplify(self.a + other.a),
                                 _simplify(self.b + other.b))

    def sub(self, other: 'RingElement') -> 'RingElement':
        _require_same_ring(self, other, 'subtraction')
        return self.ring.element(_simplify(self.a - other.a),
                                 _simplify(self.b - other.b))

    def negate(self) -> 'RingElement':
        return self.ring.element(_simplify(-self.a), _simplify(-self.b))

    def scale(self, k) -> 'RingElement':
        ks = sp.sympify(k)
        return self.ring.element(_simplify(self.a * ks),
                                 _simplify(self.b * ks))

    # --- Multiplicative structure ---------------------------------------

    def mult(self, other: 'RingElement') -> 'RingElement':
        """Ring product, applying  g² = t·g − 1.

        (a₁ + b₁·g)·(a₂ + b₂·g)
            = a₁·a₂ + (a₁·b₂ + b₁·a₂)·g + b₁·b₂·g²
            = a₁·a₂ + (a₁·b₂ + b₁·a₂)·g + b₁·b₂·(t·g − 1)
            = (a₁·a₂ − b₁·b₂) + (a₁·b₂ + b₁·a₂ + b₁·b₂·t)·g.
        """
        _require_same_ring(self, other, 'multiplication')
        a1, b1 = self.a, self.b
        a2, b2 = other.a, other.b
        new_a = _simplify(a1 * a2 - b1 * b2)
        new_b = _simplify(a1 * b2 + b1 * a2 + b1 * b2 * self.t)
        return self.ring.element(new_a, new_b)

    def conjugate(self) -> 'RingElement':
        """conj(a + b·g) = (a + b·t) − b·g.

        Galois conjugation: replaces g with its inverse  g⁻¹ = t − g.
        It's an involution on the ring.
        """
        return self.ring.element(_simplify(self.a + self.b * self.t),
                                 _simplify(-self.b))

    def norm(self) -> sp.Expr:
        """Norm  N(a + b·g) = a² + a·b·t + b² ∈ Q[t].

        This is the scalar component of  x · conj(x); the g-component
        vanishes identically. Norm is multiplicative: N(xy) = N(x)·N(y).
        For any integer power of the generator,  N(g^n) = 1.
        """
        return _simplify(self.a * self.a + self.a * self.b * self.t + self.b * self.b)

    def inverse(self) -> 'RingElement':
        """Multiplicative inverse via  x⁻¹ = conj(x) / N(x).

        Raises ZeroDivisionError when the norm simplifies to 0 (light cone).
        """
        n = self.norm()
        if _simplify(n) == 0:
            raise ZeroDivisionError(
                f'RingElement has zero norm at t={self.t}, a={self.a}, b={self.b}'
            )
        c = self.conjugate()
        return self.ring.element(_simplify(c.a / n), _simplify(c.b / n))

    def power(self, n: int) -> 'RingElement':
        """Integer power: repeated squaring. Negative n uses inverse."""
        if not isinstance(n, int):
            raise TypeError(f'power expects int, got {type(n).__name__}')
        if n == 0:
            return self.ring.one()
        if n < 0:
            return self.inverse().power(-n)
        result = self.ring.one()
        base = self
        k = n
        while k > 0:
            if k & 1:
                result = result.mult(base)
            base = base.mult(base)
            k >>= 1
        return result

    # --- Substitution / specialization ----------------------------------

    def subs(self, *args, **kwargs) -> 'RingElement':
        """Apply a sympy substitution to a, b, AND t.

        Returns an element of a (possibly new) ring with the substituted t.
        Substituting t to a Traction atom is fully supported — the new
        ring's arithmetic continues to use traction_simplify, so framework
        identities fire on the specialized t.
        """
        new_t = self.t.subs(*args, **kwargs) if hasattr(self.t, 'subs') else self.t
        new_ring = ChebyshevRing(t=_simplify(new_t))
        new_a = self.a.subs(*args, **kwargs) if hasattr(self.a, 'subs') else self.a
        new_b = self.b.subs(*args, **kwargs) if hasattr(self.b, 'subs') else self.b
        return new_ring.element(_simplify(new_a), _simplify(new_b))

    # --- Operator sugar -------------------------------------------------

    def __add__(self, other):
        if isinstance(other, RingElement):
            return self.add(other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, RingElement):
            return self.sub(other)
        return NotImplemented

    def __neg__(self):
        return self.negate()

    def __mul__(self, other):
        if isinstance(other, RingElement):
            return self.mult(other)
        try:
            ks = sp.sympify(other)
        except (sp.SympifyError, TypeError):
            return NotImplemented
        return self.scale(ks)

    def __rmul__(self, other):
        try:
            ks = sp.sympify(other)
        except (sp.SympifyError, TypeError):
            return NotImplemented
        return self.scale(ks)

    def __pow__(self, n):
        if isinstance(n, int):
            return self.power(n)
        return NotImplemented

    def __eq__(self, other) -> bool:
        if not isinstance(other, RingElement):
            return NotImplemented
        if not _equal(self.ring.t, other.ring.t):
            return False
        return _equal(self.a, other.a) and _equal(self.b, other.b)

    def __hash__(self) -> int:
        # Frozen dataclass needs a hash; coarse but safe.
        return hash((str(self.ring.t), str(self.a), str(self.b)))

    def __repr__(self) -> str:
        return f'RingElement({self.a} + ({self.b})·g | t={self.t})'
