"""Tests for jenk.free_linear — unconstrained 2x2 baseline."""

from __future__ import annotations

import sympy as sp
import pytest

from jenk.free_linear import FreeLinearLayer
from jenk.learnable_neuron import (
    LearnableParameter,
    SymbolicOptimizer,
    squared_error_loss,
)


class TestFreeLinearConstruction:

    def test_zero_init_all_four_at_zero(self):
        layer = FreeLinearLayer.zero_init()
        assert layer.matrix_values() == (0.0, 0.0, 0.0, 0.0)

    def test_parameters_returns_four(self):
        layer = FreeLinearLayer.zero_init()
        assert len(layer.parameters()) == 4

    def test_parameters_in_declaration_order(self):
        layer = FreeLinearLayer.zero_init()
        names = [p.symbol.name for p in layer.parameters()]
        assert names == ['m11', 'm12', 'm21', 'm22']

    def test_substitution_dict_uses_current_values(self):
        layer = FreeLinearLayer.zero_init()
        layer.m11.value = 1.0
        layer.m22.value = 2.0
        subs = layer.substitution_dict()
        assert subs[layer.m11.symbol] == sp.Float(1.0)
        assert subs[layer.m22.symbol] == sp.Float(2.0)


class TestFreeLinearForward:

    def test_forward_returns_symbolic_expression(self):
        layer = FreeLinearLayer.zero_init()
        a, b = sp.symbols('a b')
        out_a, out_b = layer.forward(a, b)
        # out_a = m11*a + m12*b, out_b = m21*a + m22*b
        assert out_a == layer.m11.symbol * a + layer.m12.symbol * b
        assert out_b == layer.m21.symbol * a + layer.m22.symbol * b

    def test_evaluate_substitutes_values(self):
        layer = FreeLinearLayer.zero_init()
        layer.m11.value = 2.0
        layer.m12.value = -1.0
        out_a, _ = layer.forward(3, 4)
        # 2*3 + (-1)*4 = 2
        assert layer.evaluate_float(out_a) == pytest.approx(2.0)


class TestFreeLinearTraining:

    def test_recovers_target_matrix_from_noise_free_data(self):
        # Target: M(s=0.5) = [[0, -1], [1, 0.5]].
        # With enough noise-free pairs, the 4-param free layer should recover
        # this exactly.
        s_target = 0.5
        layer = FreeLinearLayer.zero_init()

        # Generate consistent input/output pairs from the target matrix.
        inputs = [(1, 1), (1, -1), (2, 0), (0, 2), (3, 5)]
        pairs = []
        for x_a, x_b in inputs:
            target_a = -x_b
            target_b = x_a + s_target * x_b
            pairs.append(((x_a, x_b), (target_a, target_b)))

        loss = squared_error_loss(layer, pairs)
        opt = SymbolicOptimizer(layer, learning_rate=0.02)
        for _ in range(500):
            opt.step(loss)

        # Recovered matrix should be close to [[0, -1], [1, 0.5]].
        m11, m12, m21, m22 = layer.matrix_values()
        assert m11 == pytest.approx(0.0, abs=1e-3)
        assert m12 == pytest.approx(-1.0, abs=1e-3)
        assert m21 == pytest.approx(1.0, abs=1e-3)
        assert m22 == pytest.approx(0.5, abs=1e-3)

    def test_loss_decreases_each_step(self):
        layer = FreeLinearLayer.zero_init()
        pairs = [((1, 2), (-2, 1.5)), ((3, -1), (1, 2.5))]
        loss = squared_error_loss(layer, pairs)
        opt = SymbolicOptimizer(layer, learning_rate=0.02)
        before = opt.loss_value(loss)
        for _ in range(5):
            opt.step(loss)
        after = opt.loss_value(loss)
        assert after < before
