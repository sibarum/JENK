"""Tests for jenk.chain_learnable — multi-layer composition."""

from __future__ import annotations

import sympy as sp
import pytest

from jenk.chain_learnable import ChainLearnable
from jenk.learnable_neuron import (
    LearnableNeuron,
    LearnableParameter,
    SymbolicOptimizer,
    squared_error_loss,
)


class TestChainConstruction:

    def test_construct_two_layer(self):
        s1 = LearnableParameter.named('s1', 0.5)
        s2 = LearnableParameter.named('s2', -0.5)
        n1 = LearnableNeuron(s=s1, c=0, d=1)
        n2 = LearnableNeuron(s=s2, c=0, d=1)
        chain = ChainLearnable([n1, n2])
        assert chain.depth == 2

    def test_parameters_concatenated_in_order(self):
        s1 = LearnableParameter.named('s1', 0.0)
        s2 = LearnableParameter.named('s2', 0.0)
        n1 = LearnableNeuron(s=s1, c=0, d=1)
        n2 = LearnableNeuron(s=s2, c=0, d=1)
        chain = ChainLearnable([n1, n2])
        assert chain.parameters() == [s1, s2]

    def test_rejects_layer_missing_protocol(self):
        class NotALayer:
            pass
        with pytest.raises(TypeError, match='missing required method'):
            ChainLearnable([NotALayer()])


class TestChainForward:

    def test_two_identity_layers_pass_input_through(self):
        # Identity neuron: c=1, d=0 → out = (x_a, x_b) (s irrelevant since d=0).
        n1 = LearnableNeuron(s=0, c=1, d=0)
        n2 = LearnableNeuron(s=0, c=1, d=0)
        chain = ChainLearnable([n1, n2])
        a, b = sp.symbols('a b')
        out_a, out_b = chain.forward(a, b)
        assert sp.simplify(out_a - a) == 0
        assert sp.simplify(out_b - b) == 0

    def test_chain_matches_manual_composition(self):
        # n1 at s=0 (hyperbolic generator): out = (-x_b, x_a).
        # n2 at s=0:                         out = (-x_b, x_a).
        # Composition: input (a, b) → n1 → (-b, a) → n2 → (-a, -b).
        n1 = LearnableNeuron(s=0, c=0, d=1)
        n2 = LearnableNeuron(s=0, c=0, d=1)
        chain = ChainLearnable([n1, n2])
        a, b = sp.symbols('a b')
        out_a, out_b = chain.forward(a, b)
        assert sp.simplify(out_a + a) == 0
        assert sp.simplify(out_b + b) == 0


class TestChainTraining:

    def test_recovers_in_image_two_layer_teacher(self):
        # Teacher: two layers at s=0.5, c=0, d=1. Composition is in the chain image.
        # Student: same architecture, both s's learnable.
        s_target = 0.5

        # Generate teacher outputs
        from jenk.learnable_neuron import LearnableNeuron as LN
        teacher1 = LN(s=s_target, c=0, d=1)
        teacher2 = LN(s=s_target, c=0, d=1)
        teacher_chain = ChainLearnable([teacher1, teacher2])
        inputs = [(1, 1), (2, -1), (1, 0.5), (-1, 1.5), (0.5, 0.5)]
        pairs = []
        for x_a, x_b in inputs:
            out_a, out_b = teacher_chain.forward(x_a, x_b)
            pairs.append(((x_a, x_b), (float(out_a), float(out_b))))

        # Build student with both s's learnable
        s1 = LearnableParameter.named('s1', 0.0)
        s2 = LearnableParameter.named('s2', 0.0)
        student1 = LN(s=s1, c=0, d=1)
        student2 = LN(s=s2, c=0, d=1)
        student = ChainLearnable([student1, student2])

        loss = squared_error_loss(student, pairs)
        opt = SymbolicOptimizer(student, learning_rate=0.02)
        for _ in range(400):
            opt.step(loss)

        # Student should reach near-zero loss. Note: s1 and s2 may not each be
        # 0.5 — there's a permutation/equivalence ambiguity in the chain
        # decomposition. The output behavior is what matters.
        assert opt.loss_value(loss) < 1e-6
