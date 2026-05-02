"""ChebyshevChain — left-to-right composition of ChebyshevNeurons.

A chain is just a list of layers; its forward pass threads a 2-tuple
through each layer in declaration order. Each layer can have its own
trace `t` and its own weight `(c, d)` — composition is plain function
composition of 2-in/2-out maps, no trace-mixing required because data
between layers is a bare 2-tuple, not a tagged ring element.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from old.jenk.chebyshev_neuron import ChebyshevNeuron


@dataclass(frozen=True)
class ChebyshevChain:
    """Composition  layers[N-1] ∘ ... ∘ layers[1] ∘ layers[0].

    Each layer's `forward` is applied in order: the output of layer i
    is the input of layer i+1. Different layers may have different
    traces; that's fine because the data flowing between them is a
    plain (out_a, out_b) tuple, not a ring-tagged value.
    """
    layers: tuple[ChebyshevNeuron, ...]

    def __post_init__(self) -> None:
        # Coerce to a tuple so the dataclass stays hashable.
        object.__setattr__(self, 'layers', tuple(self.layers))
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, ChebyshevNeuron):
                raise TypeError(
                    f'Layer {i} is not a ChebyshevNeuron (got {type(layer).__name__})'
                )

    @property
    def depth(self) -> int:
        return len(self.layers)

    def forward(self, x_a, x_b) -> tuple[sp.Expr, sp.Expr]:
        """Apply each layer left-to-right; return final 2-tuple output."""
        for layer in self.layers:
            x_a, x_b = layer.forward(x_a, x_b)
        return x_a, x_b

    def __call__(self, x_a, x_b) -> tuple[sp.Expr, sp.Expr]:
        return self.forward(x_a, x_b)

    def with_substitution(self, *args, **kwargs) -> 'ChebyshevChain':
        """Apply a sympy substitution to every layer's parameters."""
        return ChebyshevChain(
            tuple(l.with_substitution(*args, **kwargs) for l in self.layers)
        )

    def __repr__(self) -> str:
        return f'ChebyshevChain(depth={self.depth})'
