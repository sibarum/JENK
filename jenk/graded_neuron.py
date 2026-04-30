"""GradedNeuron — Traction-native diagnostic implementation.

Parallel to jenk.symbolic_neuron.SymbolicNeuron but with two crucial differences:

  1. Parameters and intermediates are sympy expressions that may contain
     the Traction atoms `z` (Zero), `w` (Omega), `null` (Null), so framework
     identities like `0 * w = 1` and `0 ** w = -1` fire automatically as the
     forward pass evaluates.

  2. The forward pass is broken into named steps and each one is recorded
     into an optional `Trace`. Both the raw sympy expression and its
     `traction_simplify`-reduced form are captured side by side, so you can
     see where identities fired (or didn't), where Null erasures appeared,
     and where the chain produced something the framework's automatic
     reductions could not collapse further.

This is the reference + diagnostic layer. It uses the same forward formula
as the matrix-recipe SymbolicNeuron (regular representation of c + d*g in
the Chebyshev ring), but evaluated through sympy with framework atoms in
play. The point is to make the layer's behavior fully inspectable in
framework-native terms before deciding what a more efficient
representation should look like.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import sympy as sp

from jenk.trace import Trace


Pair = Tuple[sp.Expr, sp.Expr]


def _sympify(value) -> sp.Expr:
    return sp.sympify(value)


@dataclass(frozen=True)
class GradedNeuron:
    """A 2-in / 2-out neuron whose ring weight is `c + d*g` at trace `s`.

    Each of `s, c, d, bias_a, bias_b` is a sympy expression that may contain
    the Traction atoms (z, w, null). Framework identities fire automatically
    during arithmetic; no explicit reduction call is needed.

    Forward formula (same algebra as SymbolicNeuron):

        out_a = c * x_a  -  d * x_b           +  bias_a
        out_b = d * x_a  +  (c + d * s) * x_b +  bias_b

    Pass `trace=Trace(...)` to forward() to capture every intermediate.
    """

    s: sp.Expr
    c: sp.Expr
    d: sp.Expr
    bias_a: sp.Expr = sp.Integer(0)
    bias_b: sp.Expr = sp.Integer(0)

    def __post_init__(self) -> None:
        for field_name in ('s', 'c', 'd', 'bias_a', 'bias_b'):
            object.__setattr__(self, field_name, _sympify(getattr(self, field_name)))

    # --- Constructors ----------------------------------------------------

    @staticmethod
    def generator(s, bias_a=0, bias_b=0) -> 'GradedNeuron':
        """Neuron whose ring weight is the generator g (c=0, d=1) at trace s."""
        return GradedNeuron(s=s, c=sp.Integer(0), d=sp.Integer(1),
                            bias_a=bias_a, bias_b=bias_b)

    @staticmethod
    def identity(s) -> 'GradedNeuron':
        """Neuron whose ring weight is the multiplicative identity (c=1, d=0)."""
        return GradedNeuron(s=s, c=sp.Integer(1), d=sp.Integer(0))

    # --- Forward pass ----------------------------------------------------

    def forward(self, x_a, x_b, trace: Trace | None = None) -> Pair:
        """Apply the neuron to a 2-tuple input.

        When a `Trace` is provided, every named intermediate is recorded.
        Returns (out_a, out_b) as sympy expressions.
        """
        x_a = _sympify(x_a)
        x_b = _sympify(x_b)

        if trace is not None:
            trace.record('input.x_a', x_a, note='input scalar component')
            trace.record('input.x_b', x_b, note='input generator component')
            trace.record('param.s', self.s, note='trace parameter')
            trace.record('param.c', self.c, note='ring weight scalar part')
            trace.record('param.d', self.d, note='ring weight g-part')

        # out_a path: c*x_a - d*x_b + bias_a
        m1 = self.c * x_a
        if trace is not None:
            trace.record('m1 = c * x_a', m1,
                         operands={'c': self.c, 'x_a': x_a})
        m2 = self.d * x_b
        if trace is not None:
            trace.record('m2 = d * x_b', m2,
                         operands={'d': self.d, 'x_b': x_b})
        out_a_core = m1 - m2
        if trace is not None:
            trace.record('out_a_core = m1 - m2', out_a_core,
                         operands={'m1': m1, 'm2': m2},
                         note='cancellation may produce null here')
        out_a = out_a_core + self.bias_a
        if trace is not None:
            trace.record('out_a = out_a_core + bias_a', out_a,
                         operands={'out_a_core': out_a_core, 'bias_a': self.bias_a})

        # out_b path: d*x_a + (c + d*s)*x_b + bias_b
        m3 = self.d * x_a
        if trace is not None:
            trace.record('m3 = d * x_a', m3,
                         operands={'d': self.d, 'x_a': x_a})
        m4 = self.d * self.s
        if trace is not None:
            trace.record('m4 = d * s', m4,
                         operands={'d': self.d, 's': self.s},
                         note='this is where 0*w -> 1 type identities can fire')
        m5 = self.c + m4
        if trace is not None:
            trace.record('m5 = c + m4', m5,
                         operands={'c': self.c, 'm4': m4})
        m6 = m5 * x_b
        if trace is not None:
            trace.record('m6 = m5 * x_b', m6,
                         operands={'m5': m5, 'x_b': x_b})
        out_b_core = m3 + m6
        if trace is not None:
            trace.record('out_b_core = m3 + m6', out_b_core,
                         operands={'m3': m3, 'm6': m6})
        out_b = out_b_core + self.bias_b
        if trace is not None:
            trace.record('out_b = out_b_core + bias_b', out_b,
                         operands={'out_b_core': out_b_core, 'bias_b': self.bias_b})

        if trace is not None:
            trace.record('output.out_a', out_a, note='final scalar component')
            trace.record('output.out_b', out_b, note='final generator component')

        return out_a, out_b

    def __call__(self, x_a, x_b, trace: Trace | None = None) -> Pair:
        return self.forward(x_a, x_b, trace=trace)

    # --- Introspection ---------------------------------------------------

    def with_substitution(self, *args, **kwargs) -> 'GradedNeuron':
        """Apply a sympy substitution to every parameter and return a new neuron."""
        return GradedNeuron(
            s=self.s.subs(*args, **kwargs),
            c=self.c.subs(*args, **kwargs),
            d=self.d.subs(*args, **kwargs),
            bias_a=self.bias_a.subs(*args, **kwargs),
            bias_b=self.bias_b.subs(*args, **kwargs),
        )


def chain(*neurons: GradedNeuron):
    """Compose neurons left-to-right with optional trace propagation.

    chain(n1, n2)(x_a, x_b, trace=t) calls n1 then n2, both recording into t.
    """
    def composed(x_a, x_b, trace: Trace | None = None) -> Pair:
        for i, n in enumerate(neurons):
            if trace is not None:
                trace.record(f'chain.layer[{i}].entry', sp.sympify(x_a),
                             note=f'about to apply layer {i}; x_b follows')
                trace.record(f'chain.layer[{i}].entry.x_b', sp.sympify(x_b))
            x_a, x_b = n.forward(x_a, x_b, trace=trace)
        return x_a, x_b
    return composed
