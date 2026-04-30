"""ChainLearnable — compose N learnable layers as a single trainable unit.

A `ChainLearnable` wraps any number of layers that conform to the
LearnableNeuron protocol (LearnableNeuron, FreeLinearLayer, or any
duck-compatible thing). The forward pass composes the layers
left-to-right; the combined parameter list is the concatenation of
each layer's parameters in declaration order.

This conforms to the same protocol as a single layer (parameters,
parameter_symbols, substitution_dict, evaluate, evaluate_float,
forward), so SymbolicOptimizer accepts it without modification.

The motivating experiment: train a multi-layer chain student to imitate
a teacher matrix. The Stage-3 finding established that some matrices
(notably M_proj = [[0, 0], [0, 1]]) are *outside* the image of any
2-layer chain — the chain composes determinant-preserving maps up to
scaling and lands in SL(2)-flavored territory, while M_proj is rank-1
with det=0. ChainLearnable lets us train such a student and watch the
loss floor land where the theory predicts it should.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from jenk.learnable_neuron import LearnableParameter


@dataclass
class ChainLearnable:
    """Composition of N layers conforming to the learnable-layer protocol.

    Construction:
        chain = ChainLearnable([layer1, layer2, layer3])

    Forward:
        chain.forward(x_a, x_b)
        # equivalent to layer3.forward(*layer2.forward(*layer1.forward(x_a, x_b)))

    Parameter list is the concatenation of each layer's `.parameters()`,
    in layer-declaration order — so SymbolicOptimizer steps all layers
    jointly.
    """
    layers: list

    def __post_init__(self) -> None:
        # Sanity: each layer must have the expected protocol surface.
        required = ('forward', 'parameters', 'parameter_symbols')
        for i, layer in enumerate(self.layers):
            for name in required:
                if not hasattr(layer, name):
                    raise TypeError(
                        f'Layer {i} ({type(layer).__name__}) is missing required '
                        f'method {name!r}; cannot be wrapped in ChainLearnable.'
                    )

    # --- Forward ---------------------------------------------------------

    def forward(self, x_a, x_b, trace=None) -> tuple[sp.Expr, sp.Expr]:
        """Symbolic forward — composes each layer's forward in order."""
        x_a = sp.sympify(x_a)
        x_b = sp.sympify(x_b)
        for i, layer in enumerate(self.layers):
            if trace is not None:
                trace.record(f'chain.layer[{i}].entry.x_a', x_a)
                trace.record(f'chain.layer[{i}].entry.x_b', x_b)
            x_a, x_b = layer.forward(x_a, x_b, trace=trace) \
                if 'trace' in layer.forward.__code__.co_varnames \
                else layer.forward(x_a, x_b)
        return x_a, x_b

    def __call__(self, x_a, x_b, trace=None) -> tuple[sp.Expr, sp.Expr]:
        return self.forward(x_a, x_b, trace=trace)

    # --- Combined parameter view ----------------------------------------

    def parameters(self) -> list[LearnableParameter]:
        """All learnable parameters across all layers, in declaration order."""
        result = []
        for layer in self.layers:
            result.extend(layer.parameters())
        return result

    def parameter_symbols(self) -> list[sp.Symbol]:
        return [p.symbol for p in self.parameters()]

    def substitution_dict(self) -> dict:
        return {p.symbol: sp.Float(p.value) for p in self.parameters()}

    def evaluate(self, expr: sp.Expr) -> sp.Expr:
        return expr.subs(self.substitution_dict())

    def evaluate_float(self, expr: sp.Expr) -> float:
        return float(self.evaluate(expr))

    # --- Convenience -----------------------------------------------------

    @property
    def depth(self) -> int:
        return len(self.layers)
