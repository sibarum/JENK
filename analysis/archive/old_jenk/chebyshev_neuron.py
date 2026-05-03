"""ChebyshevNeuron — a 2-in / 2-out layer that is ring multiplication.

A layer is parameterized by a ring weight  w = c + d·g  at trace t, plus
optional additive biases. Its forward pass multiplies the input
2-tuple (x_a, x_b), interpreted as the ring element  x = x_a + x_b·g, by
the weight, and returns the result's (a, b) coefficients.

In other words, the layer is *literally* multiplication-by-w in the
Chebyshev ring  Q[t][g]/(g² − t·g + 1).  The implementation delegates
all algebra to `jenk.chebyshev_ring`, so there is exactly one place
where ring multiplication is defined.

Naming
------
- `t` is the Chebyshev trace parameter (g + g⁻¹ = t, g² = t·g − 1).
  Cardinals on `t` are the classical Chebyshev signature regions:
  |t| < 2 elliptic, |t| = 2 parabolic light cone, |t| > 2 hyperbolic.

- The four-mode framework parameter (the one that satisfies
  g(s) = 0^(s/2) and g(s)² = 0^s, with cardinals at s ∈ {0, 1, ω, −1})
  is a *separate* concept, by convention named `s` elsewhere. This
  module does not use `s`. See `jenk.chebyshev_ring` for the bridge
  between mode generators and their natural traces.

Construction
------------
    ChebyshevNeuron(t=0, c=0, d=1)                # elliptic 90° rotor
    ChebyshevNeuron(t=2, c=1, d=0)                # identity at hyperbolic trace
    ChebyshevNeuron(t=Omega(), c=0, d=1)          # symbolic: trace = ω atom

Forward
-------
    n = ChebyshevNeuron(t=0, c=0, d=1)
    out_a, out_b = n.forward(x_a, x_b)
    # equivalently: out_a + out_b·g  =  (c + d·g) · (x_a + x_b·g)  in the ring

Biases are added after the multiplication:
    out_a += bias_a
    out_b += bias_b
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from old.jenk.chebyshev_ring import ChebyshevRing, RingElement
from old.jenk.traction import traction_simplify


def _sympify(value) -> sp.Expr:
    return sp.sympify(value)


def _simplify(expr: sp.Expr) -> sp.Expr:
    try:
        return traction_simplify(expr)
    except Exception:
        return sp.simplify(expr)


@dataclass(frozen=True)
class ChebyshevNeuron:
    """A 2-in / 2-out layer that multiplies its input by  c + d·g  at trace t.

    Construction parameters are coerced to sympy expressions; they may be
    Python numbers, sympy symbols, or Traction atoms (Zero, Omega, Null).

    The layer's algebra is exactly ring multiplication in the Chebyshev
    ring at the given trace; see `jenk.chebyshev_ring` for the underlying
    implementation.

    Fields
    ------
    t : trace parameter (Chebyshev: g² = t·g − 1)
    c : scalar part of the ring weight
    d : g-part of the ring weight
    bias_a, bias_b : additive bias on the 2-tuple output
    """
    t: sp.Expr
    c: sp.Expr
    d: sp.Expr
    bias_a: sp.Expr = sp.Integer(0)
    bias_b: sp.Expr = sp.Integer(0)

    def __post_init__(self) -> None:
        for slot in ('t', 'c', 'd', 'bias_a', 'bias_b'):
            object.__setattr__(self, slot, _sympify(getattr(self, slot)))

    # --- Constructors --------------------------------------------------

    @staticmethod
    def identity(t) -> 'ChebyshevNeuron':
        """Layer with weight = 1 (= 1 + 0·g) at the given trace.

        Forward is the identity map: (x_a, x_b) → (x_a, x_b). The trace
        is recorded but acts as a no-op since d = 0.
        """
        return ChebyshevNeuron(t=t, c=sp.Integer(1), d=sp.Integer(0))

    @staticmethod
    def generator(t) -> 'ChebyshevNeuron':
        """Layer with weight = g (= 0 + 1·g) at the given trace.

        Forward is multiplication by the ring generator. At t=0 (elliptic
        natural trace) this is the 90° rotor: (x_a, x_b) → (−x_b, x_a).
        """
        return ChebyshevNeuron(t=t, c=sp.Integer(0), d=sp.Integer(1))

    # --- Ring view -----------------------------------------------------

    def ring(self) -> ChebyshevRing:
        """The Chebyshev ring this layer operates in."""
        return ChebyshevRing(t=self.t)

    def weight(self) -> RingElement:
        """The ring element  c + d·g  this layer multiplies by."""
        return self.ring().element(self.c, self.d)

    # --- Forward pass --------------------------------------------------

    def forward(self, x_a, x_b) -> tuple[sp.Expr, sp.Expr]:
        """Apply  out = w · x + bias,  where w and x are ring elements.

        Returns the (a, b) coefficients of the result as a 2-tuple.
        """
        x = self.ring().element(x_a, x_b)
        result = self.weight() * x  # delegated to RingElement.mult, which traction_simplifies
        out_a = _simplify(result.a + self.bias_a)
        out_b = _simplify(result.b + self.bias_b)
        return out_a, out_b

    def __call__(self, x_a, x_b) -> tuple[sp.Expr, sp.Expr]:
        return self.forward(x_a, x_b)

    # --- Substitution / specialization --------------------------------

    def with_substitution(self, *args, **kwargs) -> 'ChebyshevNeuron':
        """Apply a sympy substitution to every field; return a new neuron.

        Useful for partial evaluation — e.g., a neuron defined with a
        free symbol `t_sym` can be specialized to t_sym = 0.
        """
        def subs(expr):
            return expr.subs(*args, **kwargs) if hasattr(expr, 'subs') else expr
        return ChebyshevNeuron(
            t=_simplify(subs(self.t)),
            c=_simplify(subs(self.c)),
            d=_simplify(subs(self.d)),
            bias_a=_simplify(subs(self.bias_a)),
            bias_b=_simplify(subs(self.bias_b)),
        )

    def __repr__(self) -> str:
        bias = ''
        if self.bias_a != 0 or self.bias_b != 0:
            bias = f', bias=({self.bias_a}, {self.bias_b})'
        return f'ChebyshevNeuron(t={self.t}, c={self.c}, d={self.d}{bias})'
