"""Symbolic prototype neuron — the regular-representation matrix as a recipe.

This is a research prototype, not a production layer. Every parameter
and intermediate is a sympy expression (which may contain `omega` from
jenk.graded). Nothing is ever projected to a float unless the user
explicitly substitutes. The goal is to make the projective and elliptic
modes (cardinals at s=-1 and s=omega) operationally testable without
having to commit to any numerical proxy or projection rule.

Each neuron stores five sympy expressions:

  s, c, d:           the "weight" is the ring element  c + d*g  at trace s
  bias_a, bias_b:    additive bias on the 2D output

The forward pass applies the regular-representation matrix M(c, d, s)
of the Chebyshev ring  Q(omega)[s][g] / (g^2 - s*g + 1):

  M(c, d, s) = [[c,   -d       ],
                [d,   c + d*s  ]]

  out_a = c * x_a  -  d * x_b           +  bias_a
  out_b = d * x_a  +  (c + d*s) * x_b   +  bias_b

Inter-neuron data is a plain 2-tuple (out_a, out_b) of sympy expressions
with no `s` tag. Each neuron's `s` is internal to its own matrix, so
chaining neurons with different `s` values is just function composition
of 2x2 matrices -- no s-mixing required at this layer. (The alternative,
where data flows tagged with `s` and composition has to handle s-mixing,
is a separate variant deferred per discussion.)

This module deliberately does NOT include activations, learning, layers,
or batching. It is the simplest possible thing that lets one neuron
operate at any `s` -- including symbolic `s = omega` -- and produce an
unambiguous symbolic output you can inspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import sympy as sp

from jenk.graded import Graded


def _sympify(value) -> sp.Expr:
    return sp.sympify(value)


# Type alias for the 2-tuple input/output shape.
Pair = Tuple[sp.Expr, sp.Expr]


@dataclass(frozen=True)
class SymbolicNeuron:
    """A single 2-in / 2-out neuron whose matrix is the ring's regular rep.

    All fields are coerced to sympy expressions in __post_init__, so
    Python ints/floats and bare sympy symbols can be passed directly.
    """
    s: sp.Expr
    c: sp.Expr
    d: sp.Expr
    bias_a: sp.Expr = sp.Integer(0)
    bias_b: sp.Expr = sp.Integer(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, 's', _sympify(self.s))
        object.__setattr__(self, 'c', _sympify(self.c))
        object.__setattr__(self, 'd', _sympify(self.d))
        object.__setattr__(self, 'bias_a', _sympify(self.bias_a))
        object.__setattr__(self, 'bias_b', _sympify(self.bias_b))

    # --- Constructors ----------------------------------------------------

    @staticmethod
    def generator(s, bias_a=0, bias_b=0) -> 'SymbolicNeuron':
        """Neuron whose ring weight is the generator g (c=0, d=1) at trace s.

        This is the minimal one-parameter neuron: matrix is
            M(0, 1, s) = [[0, -1], [1, s]],
        giving  out_a = -x_b,  out_b = x_a + s*x_b.
        """
        return SymbolicNeuron(s=s, c=sp.Integer(0), d=sp.Integer(1),
                              bias_a=bias_a, bias_b=bias_b)

    @staticmethod
    def identity(s) -> 'SymbolicNeuron':
        """Neuron whose ring weight is the multiplicative identity (c=1, d=0).

        Matrix is the 2x2 identity; trace is recorded but acts as no-op.
        Useful as a sanity baseline.
        """
        return SymbolicNeuron(s=s, c=sp.Integer(1), d=sp.Integer(0))

    # --- Forward pass ----------------------------------------------------

    def matrix(self) -> sp.Matrix:
        """The 2x2 regular-representation matrix M(c, d, s)."""
        return sp.Matrix([
            [self.c, -self.d],
            [self.d, self.c + self.d * self.s],
        ])

    def forward(self, x_a, x_b) -> Pair:
        """Apply the neuron to a 2-tuple input. Returns (out_a, out_b)."""
        x_a = _sympify(x_a)
        x_b = _sympify(x_b)
        out_a = self.c * x_a - self.d * x_b + self.bias_a
        out_b = self.d * x_a + (self.c + self.d * self.s) * x_b + self.bias_b
        return out_a, out_b

    def __call__(self, x_a, x_b) -> Pair:
        return self.forward(x_a, x_b)

    # --- Probes / introspection ------------------------------------------

    def carrier_g_squared(self) -> Graded:
        """Return the *carrier-side* target  g(s)^2 = (1, s) reduced.

        This is the framework's cardinal target at this neuron's `s`. It
        is NOT in general equal to anything the matrix M(s) computes -- the
        ring-g (where M lives) and the carrier-g (where (1, s/2)^2 = (1, s))
        are distinct objects sharing the symbol `g`. Exposing it here lets
        callers compare the two sides without the algebra silently
        conflating them.
        """
        return Graded(sp.Integer(1), self.s).reduced()

    def with_substitution(self, *args, **kwargs) -> 'SymbolicNeuron':
        """Apply a sympy substitution to every parameter and return a new neuron.

        Useful for partial evaluation -- e.g., a neuron defined with
        a free symbol `s_sym` can be specialized to s_sym=omega.
        """
        return SymbolicNeuron(
            s=self.s.subs(*args, **kwargs),
            c=self.c.subs(*args, **kwargs),
            d=self.d.subs(*args, **kwargs),
            bias_a=self.bias_a.subs(*args, **kwargs),
            bias_b=self.bias_b.subs(*args, **kwargs),
        )


def chain(*neurons: SymbolicNeuron):
    """Compose neurons left-to-right: chain(n1, n2)(x) = n2(n1(x)).

    Returns a callable taking (x_a, x_b) and returning (out_a, out_b).
    Inter-neuron data is the plain 2-tuple, so different `s` values across
    the chain are fine -- each neuron's `s` only affects its own matrix.
    """
    def composed(x_a, x_b) -> Pair:
        for n in neurons:
            x_a, x_b = n.forward(x_a, x_b)
        return x_a, x_b
    return composed
