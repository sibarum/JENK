"""Graded projective basis with cyclic squaring rule, plus the v1 neuron.

Basis  k_0, ..., k_{N-1}  with multiplication rules

    k_i · k_i  =  k_{(i + 1) mod N}     (squaring; James's rule)
    k_i · k_j  =  k_{(2i + j) mod N}    (mixed products, i ≠ j)

The mixed-product rule is asymmetric — k_0·k_1 = k_1 but k_1·k_0 = k_2 — so
the algebra is non-commutative.

A ``Projective(N)`` value is a length-N coefficient vector over the reals.
All arithmetic stays in raw numpy / python — sympy does not cross this
boundary.

Encoding
--------
Four slots ``(in_0, op_0, in_1, out_0) ↦ (k_0, k_1, k_2, k_3)``.
``in_0`` and ``in_1`` are the two operands; ``op_0`` is an integer enum
(set to 0 for v1 multiplication training); ``out_0`` is initialised to 0
in inputs and is read out from the network output as the prediction by
default.

The v1 neuron (``SquaredInputProjectiveNeuron``) has one bias (also a
4-slot Projective).  Forward is ``output = (input · input) · bias``;
prediction is ``output[readout_slot]`` (default ``OUT_0`` = k_3); loss is
squared error; gradients are computed analytically from the bilinear
product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np


# Slot conventions for the v0 four-slot encoding
IN_0, OP_0, IN_1, OUT_0 = 0, 1, 2, 3


def basis_product_index(i: int, j: int, n: int) -> int:
    """Index k where  k_i · k_j = k_k  under the cyclic-squaring algebra."""
    if i == j:
        return (i + 1) % n
    return (2 * i + j) % n


@dataclass
class Projective:
    """A coefficient vector in the N-dimensional graded projective basis."""
    coeffs: np.ndarray

    def __post_init__(self) -> None:
        self.coeffs = np.asarray(self.coeffs, dtype=float)

    @property
    def n(self) -> int:
        return len(self.coeffs)

    @staticmethod
    def zeros(n: int) -> "Projective":
        return Projective(np.zeros(n))

    def __getitem__(self, idx: int) -> float:
        return float(self.coeffs[idx])

    def __add__(self, other: "Projective") -> "Projective":
        return Projective(self.coeffs + other.coeffs)

    def __sub__(self, other: "Projective") -> "Projective":
        return Projective(self.coeffs - other.coeffs)

    def __mul__(self, other: Union["Projective", float, int]) -> "Projective":
        if isinstance(other, Projective):
            assert self.n == other.n, f"basis size mismatch: {self.n} vs {other.n}"
            n = self.n
            out = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    k = basis_product_index(i, j, n)
                    out[k] += self.coeffs[i] * other.coeffs[j]
            return Projective(out)
        return Projective(self.coeffs * float(other))

    def __rmul__(self, other) -> "Projective":
        return Projective(self.coeffs * float(other))


def encode_problem(a: float, op: int, b: float, n: int = 4) -> Projective:
    """Encode  (a, op, b)  into a 4-slot Projective with ``out_0`` set to 0."""
    coeffs = np.zeros(n)
    coeffs[IN_0] = a
    coeffs[OP_0] = op
    coeffs[IN_1] = b
    return Projective(coeffs)


@dataclass
class SquaredInputProjectiveNeuron:
    """v1: forward = (input · input) · bias, prediction read from ``readout_slot``.

    The first product squares the input *vector* using the algebra's
    ``k_n²`` rule, which surfaces quadratic features (``a²``, ``a·b``,
    ``b²``) in the hidden layer.  The second product reads them out
    against the trainable ``bias``.  Net result: function class is
    quadratic in the input — enough to fit ``a·b`` exactly.

    ``readout_slot`` selects which slot of ``output = hidden · bias``
    is treated as the prediction.  Default ``OUT_0`` (k_3) reads the
    rigid-direction slot whose (g, h) gradient is always (a·b, 2·a·b);
    setting ``readout_slot=0`` reads the input-dependent-direction
    slot whose gradient is (a·b, a²+b², b²) and avoids the v1
    convergence degeneracy.

    Caveat (single-op v1): with the v0 four-slot encoding and a
    *non-zero constant* ``op`` value ``c``, the squared input picks up
    ``c·a``, ``c·b``, ``c²`` terms that can't all be zeroed at the
    output without losing the ``a·b`` coefficient.  So v1 trains with
    ``op = 0``.
    """

    bias: Projective
    readout_slot: int = OUT_0

    @staticmethod
    def zeros(n: int = 4, readout_slot: int = OUT_0) -> "SquaredInputProjectiveNeuron":
        return SquaredInputProjectiveNeuron(
            bias=Projective.zeros(n), readout_slot=readout_slot
        )

    def feedforward(self, input_: Projective) -> Projective:
        hidden = input_ * input_
        return hidden * self.bias

    def predict(self, input_: Projective) -> float:
        return self.feedforward(input_)[self.readout_slot]

    def gradients(self, input_: Projective, target: float) -> np.ndarray:
        """``dL / d(bias_j)`` for ``L = (output[readout_slot] − target) ** 2``.

        The squaring step depends only on ``input``, not ``bias``, so
        the gradient w.r.t. ``bias`` is the same shape as v0's, just
        with ``hidden = input · input`` substituted for ``input`` in
        the contributor sum.
        """
        n = input_.n
        hidden = input_ * input_
        residual = (hidden * self.bias)[self.readout_slot] - target
        d_pred = np.zeros(n)
        for j in range(n):
            for i in range(n):
                if basis_product_index(i, j, n) == self.readout_slot:
                    d_pred[j] += hidden.coeffs[i]
        return 2.0 * residual * d_pred

    def update(self, grad: np.ndarray, lr: float) -> None:
        self.bias.coeffs -= lr * grad

    def step(self, input_: Projective, target: float, lr: float) -> float:
        pred = self.predict(input_)
        grad = self.gradients(input_, target)
        self.update(grad, lr)
        return pred
