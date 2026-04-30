"""FreeLinearLayer — unconstrained 2x2 linear baseline.

A four-parameter linear layer with no structural constraint:

    out_a = m11 * x_a + m12 * x_b
    out_b = m21 * x_a + m22 * x_b

Used as the inductive-bias baseline against LearnableNeuron. The
hypothesis under test: LearnableNeuron, with only `s` learnable and
`c=0, d=1` fixed, should generalize better than FreeLinearLayer on data
generated from the framework's ring structure — because it can't
overfit the noise into matrix entries that don't belong to the ring
image. FreeLinearLayer has four free entries and no such bias.

This layer satisfies the same protocol as LearnableNeuron (parameters,
parameter_symbols, substitution_dict, evaluate, evaluate_float, forward),
so SymbolicOptimizer accepts it via duck typing — no changes needed
on the optimizer side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from jenk.learnable_neuron import LearnableParameter


@dataclass
class FreeLinearLayer:
    """A 2x2 linear layer with all four entries learnable.

    Construction takes four LearnableParameters (m11, m12, m21, m22) for
    the matrix entries. Forward returns sympy expressions in those
    parameter symbols.
    """
    m11: LearnableParameter
    m12: LearnableParameter
    m21: LearnableParameter
    m22: LearnableParameter

    _all_params: list[LearnableParameter] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._all_params = [self.m11, self.m12, self.m21, self.m22]

    @staticmethod
    def zero_init() -> 'FreeLinearLayer':
        """Construct with all four entries initialized to 0."""
        return FreeLinearLayer(
            m11=LearnableParameter.named('m11', 0.0),
            m12=LearnableParameter.named('m12', 0.0),
            m21=LearnableParameter.named('m21', 0.0),
            m22=LearnableParameter.named('m22', 0.0),
        )

    # --- Forward ---------------------------------------------------------

    def forward(self, x_a, x_b, trace=None) -> tuple[sp.Expr, sp.Expr]:
        """Symbolic forward — returns sympy expressions in the parameter symbols."""
        x_a = sp.sympify(x_a)
        x_b = sp.sympify(x_b)
        out_a = self.m11.symbol * x_a + self.m12.symbol * x_b
        out_b = self.m21.symbol * x_a + self.m22.symbol * x_b
        if trace is not None:
            trace.record('input.x_a', x_a)
            trace.record('input.x_b', x_b)
            trace.record('out_a', out_a)
            trace.record('out_b', out_b)
        return out_a, out_b

    def __call__(self, x_a, x_b, trace=None) -> tuple[sp.Expr, sp.Expr]:
        return self.forward(x_a, x_b, trace=trace)

    # --- Same protocol as LearnableNeuron --------------------------------

    def parameters(self) -> list[LearnableParameter]:
        return list(self._all_params)

    def parameter_symbols(self) -> list[sp.Symbol]:
        return [p.symbol for p in self._all_params]

    def substitution_dict(self) -> dict:
        return {p.symbol: sp.Float(p.value) for p in self._all_params}

    def evaluate(self, expr: sp.Expr) -> sp.Expr:
        return expr.subs(self.substitution_dict())

    def evaluate_float(self, expr: sp.Expr) -> float:
        return float(self.evaluate(expr))

    # --- Convenience -----------------------------------------------------

    def matrix_values(self) -> tuple[float, float, float, float]:
        """Return (m11, m12, m21, m22) at their current numeric values."""
        return (self.m11.value, self.m12.value, self.m21.value, self.m22.value)
