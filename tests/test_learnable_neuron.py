"""Tests for jenk.learnable_neuron — symbolic-first gradient training."""

from __future__ import annotations

import math
import sympy as sp
import pytest

from jenk.learnable_neuron import (
    LearnableParameter,
    LearnableNeuron,
    SymbolicOptimizer,
    squared_error_loss,
)
from jenk.traction import z, w


# =============================================================================
# LearnableParameter
# =============================================================================

class TestLearnableParameter:

    def test_symbol_and_value_are_separate(self):
        p = LearnableParameter(sp.Symbol('s'), 0.5)
        assert isinstance(p.symbol, sp.Symbol)
        assert p.value == 0.5

    def test_named_constructor(self):
        p = LearnableParameter.named('s_param', 1.5)
        assert str(p.symbol) == 's_param'
        assert p.value == 1.5

    def test_named_creates_real_symbol_by_default(self):
        p = LearnableParameter.named('s', 0.0)
        # Real symbols compare equal to themselves under is_real
        assert p.symbol.is_real is True

    def test_value_is_mutable(self):
        p = LearnableParameter.named('s', 0.0)
        p.value = 2.0
        assert p.value == 2.0


# =============================================================================
# LearnableNeuron — registration and forward
# =============================================================================

class TestLearnableNeuronRegistration:

    def test_no_learnable_params_when_all_fixed(self):
        n = LearnableNeuron(s=0, c=0, d=1)
        assert n.parameters() == []

    def test_one_learnable_param_registered(self):
        s = LearnableParameter.named('s', 0.5)
        n = LearnableNeuron(s=s, c=0, d=1)
        assert n.parameters() == [s]

    def test_multiple_learnable_params_in_declaration_order(self):
        s = LearnableParameter.named('s', 0.5)
        c = LearnableParameter.named('c', 0.0)
        d = LearnableParameter.named('d', 1.0)
        n = LearnableNeuron(s=s, c=c, d=d)
        assert n.parameters() == [s, c, d]

    def test_substitution_dict_uses_current_values(self):
        s = LearnableParameter.named('s', 0.5)
        n = LearnableNeuron(s=s, c=0, d=1)
        subs = n.substitution_dict()
        assert subs == {s.symbol: sp.Float(0.5)}

    def test_substitution_dict_tracks_value_updates(self):
        s = LearnableParameter.named('s', 0.5)
        n = LearnableNeuron(s=s, c=0, d=1)
        s.value = 2.0
        subs = n.substitution_dict()
        assert subs == {s.symbol: sp.Float(2.0)}


class TestLearnableNeuronForward:

    def test_forward_returns_symbolic_in_parameter(self):
        s = LearnableParameter.named('s', 0.5)
        n = LearnableNeuron(s=s, c=0, d=1)
        out_a, out_b = n.forward(sp.Symbol('a'), sp.Symbol('b'))
        # Forward formula: out_a = c*x_a - d*x_b = -b, out_b = d*x_a + (c+d*s)*x_b = a + s*b
        assert out_a == -sp.Symbol('b')
        assert out_b - (sp.Symbol('a') + s.symbol * sp.Symbol('b')) == 0

    def test_evaluate_substitutes_current_value(self):
        s = LearnableParameter.named('s', 2.0)
        n = LearnableNeuron(s=s, c=0, d=1)
        _, out_b = n.forward(3, 5)
        # symbolic out_b = 3 + 5*s
        # at s=2.0: 13.0
        assert n.evaluate_float(out_b) == pytest.approx(13.0)

    def test_evaluate_picks_up_value_change(self):
        s = LearnableParameter.named('s', 1.0)
        n = LearnableNeuron(s=s, c=0, d=1)
        _, out_b = n.forward(3, 5)
        s.value = 0.0
        assert n.evaluate_float(out_b) == pytest.approx(3.0)

    def test_fixed_atoms_pass_through(self):
        # c=0, d=z (Traction zero), s=w — the framework recipe
        n = LearnableNeuron(s=w, c=0, d=z)
        out_a, out_b = n.forward(sp.Symbol('a'), sp.Symbol('b'))
        # m4 = d*s = z*w should fire to 1, so (c + d*s) = 1, out_b = z*a + b
        # The framework atom firings happen during construction; verify out_b's
        # coefficient on x_b is exactly 1 (not w or something).
        # out_b = z*a + b
        assert out_b.coeff(sp.Symbol('b')) == 1

    def test_no_learnable_params_means_empty_substitution(self):
        n = LearnableNeuron(s=0, c=0, d=1)
        assert n.substitution_dict() == {}


# =============================================================================
# SymbolicOptimizer — gradient and step
# =============================================================================

class TestSymbolicOptimizer:

    def test_loss_value_at_target_is_zero(self):
        # Build a layer with a learnable s, target s_value=0.
        # If we set s.value = 0 directly, loss should be 0.
        s = LearnableParameter.named('s', 0.0)
        n = LearnableNeuron(s=s, c=0, d=1)
        # At s=0: out = (-b, a). Pick a target consistent with that.
        loss = squared_error_loss(n, [((1, 2), (-2, 1))])
        opt = SymbolicOptimizer(n)
        assert opt.loss_value(loss) == pytest.approx(0.0, abs=1e-12)

    def test_loss_value_nonzero_when_off_target(self):
        s = LearnableParameter.named('s', 1.0)
        n = LearnableNeuron(s=s, c=0, d=1)
        loss = squared_error_loss(n, [((1, 2), (-2, 1))])
        opt = SymbolicOptimizer(n)
        assert opt.loss_value(loss) > 0

    def test_gradient_expr_returns_symbolic(self):
        s = LearnableParameter.named('s', 1.0)
        n = LearnableNeuron(s=s, c=0, d=1)
        loss = squared_error_loss(n, [((1, 2), (-2, 1))])
        opt = SymbolicOptimizer(n)
        grad = opt.gradient_expr(loss, s)
        # Gradient is a sympy expression in s.symbol
        assert isinstance(grad, sp.Expr)
        assert s.symbol in grad.free_symbols

    def test_step_decreases_loss(self):
        s = LearnableParameter.named('s', 1.0)  # off-target
        n = LearnableNeuron(s=s, c=0, d=1)
        loss = squared_error_loss(n, [((1, 2), (-2, 1))])  # target s = 0
        opt = SymbolicOptimizer(n, learning_rate=0.05)
        before = opt.loss_value(loss)
        opt.step(loss)
        after = opt.loss_value(loss)
        assert after < before

    def test_converges_to_known_target_hyperbolic(self):
        # Target s=0 (hyperbolic cardinal). Generate consistent input/output pairs.
        # At s=0, c=0, d=1: out = (-b, a) for any (a, b).
        s = LearnableParameter.named('s', 1.5)
        n = LearnableNeuron(s=s, c=0, d=1)
        pairs = [((1, 2), (-2, 1)), ((3, 5), (-5, 3)), ((-1, 4), (-4, -1))]
        loss = squared_error_loss(n, pairs)
        opt = SymbolicOptimizer(n, learning_rate=0.02)
        for _ in range(200):
            opt.step(loss)
        assert s.value == pytest.approx(0.0, abs=1e-3)
        assert opt.loss_value(loss) == pytest.approx(0.0, abs=1e-6)

    def test_converges_to_known_target_parabolic(self):
        # Target s=1 (parabolic cardinal). At s=1, c=0, d=1: out = (-b, a + b).
        s = LearnableParameter.named('s', -0.5)
        n = LearnableNeuron(s=s, c=0, d=1)
        pairs = [((1, 2), (-2, 3)), ((3, 5), (-5, 8)), ((-1, 4), (-4, 3))]
        loss = squared_error_loss(n, pairs)
        opt = SymbolicOptimizer(n, learning_rate=0.02)
        for _ in range(200):
            opt.step(loss)
        assert s.value == pytest.approx(1.0, abs=1e-3)

    def test_converges_to_known_target_elliptic(self):
        # Target s=-1 (elliptic cardinal). At s=-1, c=0, d=1: out = (-b, a - b).
        s = LearnableParameter.named('s', 0.5)
        n = LearnableNeuron(s=s, c=0, d=1)
        pairs = [((1, 2), (-2, -1)), ((3, 5), (-5, -2)), ((4, 1), (-1, 3))]
        loss = squared_error_loss(n, pairs)
        opt = SymbolicOptimizer(n, learning_rate=0.02)
        for _ in range(200):
            opt.step(loss)
        assert s.value == pytest.approx(-1.0, abs=1e-3)

    def test_lambdify_caches_across_steps(self):
        # Repeated step calls with the same loss expression should reuse the
        # compiled gradient, not re-lambdify each time.
        s = LearnableParameter.named('s', 0.5)
        n = LearnableNeuron(s=s, c=0, d=1)
        loss = squared_error_loss(n, [((1, 2), (-2, 1))])
        opt = SymbolicOptimizer(n)
        opt.step(loss)
        first_loss_lambda = opt._loss_lambda
        opt.step(loss)
        assert opt._loss_lambda is first_loss_lambda


# =============================================================================
# Multi-parameter convergence
# =============================================================================

class TestMultiParameterTraining:

    def test_two_parameters_converge_jointly(self):
        # Learn both s and c simultaneously. Target: s=0, c=2.
        # At s=0, c=2, d=1: out_a = 2*x_a - x_b, out_b = x_a + 2*x_b.
        # Use small-scale inputs to keep gradient magnitudes tame.
        s = LearnableParameter.named('s', 1.0)
        c = LearnableParameter.named('c', 0.0)
        n = LearnableNeuron(s=s, c=c, d=1)
        pairs = [
            ((1, 1), (1, 3)),
            ((1, -1), (3, -1)),
        ]
        loss = squared_error_loss(n, pairs)
        opt = SymbolicOptimizer(n, learning_rate=0.05)
        for _ in range(400):
            opt.step(loss)
        assert s.value == pytest.approx(0.0, abs=1e-2)
        assert c.value == pytest.approx(2.0, abs=1e-2)
