"""Tests for jenk.chebyshev_neuron and jenk.chebyshev_chain."""

from __future__ import annotations

import sympy as sp

from old.jenk.chebyshev_neuron import ChebyshevNeuron
from old.jenk.chebyshev_chain import ChebyshevChain
from old.jenk.chebyshev_ring import ChebyshevRing
from old.jenk.traction import Zero, Omega

a, b = sp.symbols('a b')


# --- Single-layer construction & basic algebra ---------------------------

class TestNeuronConstruction:

    def test_construct_with_numeric_params(self):
        n = ChebyshevNeuron(t=0, c=0, d=1)
        assert n.t == 0
        assert n.c == 0
        assert n.d == 1
        assert n.bias_a == 0
        assert n.bias_b == 0

    def test_construct_with_traction_atoms(self):
        n = ChebyshevNeuron(t=Omega(), c=Zero(), d=1)
        assert n.t == Omega()
        assert n.c == Zero()

    def test_identity_constructor(self):
        n = ChebyshevNeuron.identity(t=0)
        assert n.c == 1
        assert n.d == 0

    def test_generator_constructor(self):
        n = ChebyshevNeuron.generator(t=0)
        assert n.c == 0
        assert n.d == 1


# --- Forward at each cardinal trace --------------------------------------

class TestForwardAtCardinals:
    """At each natural Chebyshev trace, the generator layer realizes the
    corresponding mode action on the input pair."""

    def test_identity_layer_is_passthrough(self):
        """c=1, d=0 multiplies by the multiplicative identity. Trace is irrelevant."""
        for t_val in [0, 1, sp.Rational(1, 2), Omega()]:
            n = ChebyshevNeuron.identity(t=t_val)
            out_a, out_b = n.forward(a, b)
            assert sp.simplify(out_a - a) == 0
            assert sp.simplify(out_b - b) == 0

    def test_elliptic_generator_is_90_deg_rotation(self):
        """At t=0, the generator layer is multiplication by g (= i in projection):
        (x_a + x_b·g) · g = x_a·g + x_b·g² = x_a·g + x_b·(0·g − 1) = −x_b + x_a·g.
        So output is (−x_b, x_a)."""
        n = ChebyshevNeuron.generator(t=0)
        out_a, out_b = n.forward(a, b)
        assert sp.simplify(out_a + b) == 0
        assert sp.simplify(out_b - a) == 0

    def test_hyperbolic_trace_generator(self):
        """At t=2 (hyperbolic generator's natural trace, projected from j → 1):
        out_a = 0·x_a − 1·x_b = −x_b
        out_b = 0·x_b + 1·x_a + 1·x_b·2 = x_a + 2·x_b."""
        n = ChebyshevNeuron.generator(t=2)
        out_a, out_b = n.forward(a, b)
        assert sp.simplify(out_a + b) == 0
        assert sp.simplify(out_b - (a + 2 * b)) == 0

    def test_par_proj_shared_trace_generator(self):
        """At t = ε + k (parabolic ↔ projective shared natural trace):
        out_a = −x_b
        out_b = x_a + (ε + k)·x_b."""
        eps = Zero() ** sp.Rational(1, 2)
        k = Zero() ** sp.Rational(-1, 2)
        t = eps + k
        n = ChebyshevNeuron.generator(t=t)
        out_a, out_b = n.forward(a, b)
        assert sp.simplify(out_a + b) == 0
        assert sp.simplify(out_b - (a + t * b)) == 0

    def test_atom_omega_trace_stays_symbolic(self):
        """At t = ω (Traction atom), generator layer output keeps ω symbolic."""
        n = ChebyshevNeuron.generator(t=Omega())
        out_a, out_b = n.forward(a, b)
        # out_b = a + ω·b
        assert sp.simplify(out_a + b) == 0
        assert sp.simplify(out_b - (a + Omega() * b)) == 0


# --- Bias -----------------------------------------------------------------

class TestBias:

    def test_bias_added_to_output(self):
        n = ChebyshevNeuron(t=0, c=1, d=0, bias_a=5, bias_b=-3)
        out_a, out_b = n.forward(a, b)
        # identity weight; output = (a + 5, b − 3)
        assert sp.simplify(out_a - (a + 5)) == 0
        assert sp.simplify(out_b - (b - 3)) == 0


# --- Atoms in c, d --------------------------------------------------------

class TestAtomsInWeight:
    """Framework identities should fire when atoms appear in c or d."""

    def test_zero_in_d_with_omega_in_input(self):
        """At any t, c=1, d=Zero(): the d·x_b·t term collapses if x_b·t resolves
        to ω-shaped quantities. With x_b=Omega(), t=1: d·x_b·t = 0·ω·1 = 1."""
        n = ChebyshevNeuron(t=1, c=1, d=Zero())
        out_a, out_b = n.forward(0, Omega())
        # out_a = 1·0 − 0·ω = 0 (Zero atom in subtraction)
        # out_b = 1·ω + 0·0 + 0·ω·1 = ω + 0 + 1 = ω + 1
        # The framework identity 0·ω → 1 fires inside d·x_b·t = 0·ω·1
        assert sp.simplify(out_b - (Omega() + 1)) == 0


# --- Substitution / specialization ---------------------------------------

class TestSubstitution:

    def test_specialize_free_t_to_numeric(self):
        t_sym = sp.Symbol('t_sym')
        n = ChebyshevNeuron.generator(t=t_sym)
        n_at_zero = n.with_substitution(t_sym, 0)
        out_a, out_b = n_at_zero.forward(a, b)
        # Specialized to t=0: behaves as elliptic generator (90° rotor)
        assert sp.simplify(out_a + b) == 0
        assert sp.simplify(out_b - a) == 0

    def test_specialize_with_atom(self):
        t_sym = sp.Symbol('t_sym')
        n = ChebyshevNeuron.generator(t=t_sym)
        n_at_omega = n.with_substitution(t_sym, Omega())
        assert n_at_omega.t == Omega()


# --- Chain construction ---------------------------------------------------

class TestChainConstruction:

    def test_construct_two_layer_chain(self):
        n1 = ChebyshevNeuron.generator(t=0)
        n2 = ChebyshevNeuron.generator(t=0)
        chain = ChebyshevChain([n1, n2])
        assert chain.depth == 2

    def test_rejects_non_neuron_layer(self):
        import pytest
        with pytest.raises(TypeError):
            ChebyshevChain([ChebyshevNeuron.generator(t=0), 'not a layer'])


# --- Chain forward --------------------------------------------------------

class TestChainForward:

    def test_two_identity_layers_passthrough(self):
        n1 = ChebyshevNeuron.identity(t=0)
        n2 = ChebyshevNeuron.identity(t=1)
        chain = ChebyshevChain([n1, n2])
        out_a, out_b = chain.forward(a, b)
        assert sp.simplify(out_a - a) == 0
        assert sp.simplify(out_b - b) == 0

    def test_two_elliptic_generators_is_minus_identity(self):
        """g·g at t=0 is g² = −1, the 180° rotation. So two 90° rotations
        return (−x_a, −x_b)."""
        n1 = ChebyshevNeuron.generator(t=0)
        n2 = ChebyshevNeuron.generator(t=0)
        chain = ChebyshevChain([n1, n2])
        out_a, out_b = chain.forward(a, b)
        assert sp.simplify(out_a + a) == 0
        assert sp.simplify(out_b + b) == 0

    def test_four_elliptic_generators_returns_to_identity(self):
        """g⁴ = 1 at t=0 (since g²=−1). Four 90° rotations = identity."""
        layers = [ChebyshevNeuron.generator(t=0) for _ in range(4)]
        chain = ChebyshevChain(layers)
        out_a, out_b = chain.forward(a, b)
        assert sp.simplify(out_a - a) == 0
        assert sp.simplify(out_b - b) == 0

    def test_layers_can_have_different_traces(self):
        """No trace-matching constraint between layers; data flows as plain 2-tuple."""
        n1 = ChebyshevNeuron.generator(t=0)
        n2 = ChebyshevNeuron.generator(t=Omega())
        chain = ChebyshevChain([n1, n2])
        # n1: (a, b) → (−b, a) at t=0
        # n2: (−b, a) → ? at t=ω
        #     out_a = 0·(−b) − 1·a = −a
        #     out_b = 1·(−b) + 0·a + 1·a·ω = −b + ω·a
        out_a, out_b = chain.forward(a, b)
        assert sp.simplify(out_a + a) == 0
        assert sp.simplify(out_b - (-b + Omega() * a)) == 0


# --- Chain matches direct ring multiplication when traces align ---------

class TestChainRingAgreement:
    """A two-layer chain at the same trace t equals multiplication by the
    product of the two ring weights at t."""

    def test_two_layer_at_same_trace_equals_product(self):
        t_sym = sp.Symbol('t_sym')
        n1 = ChebyshevNeuron(t=t_sym, c=2, d=3)
        n2 = ChebyshevNeuron(t=t_sym, c=5, d=-1)
        chain = ChebyshevChain([n1, n2])
        out_a, out_b = chain.forward(a, b)

        # Direct ring product: (5 + (−1)·g) · (2 + 3·g) at trace t_sym
        # (note layer order: n1 applies first, so weight applied second is n2's)
        R = ChebyshevRing(t=t_sym)
        w1 = R.element(2, 3)
        w2 = R.element(5, -1)
        x = R.element(a, b)
        # chain: x → n1.forward(x) = w1·x; → n2.forward = w2·(w1·x) = (w2·w1)·x
        product_weight = w2 * w1
        expected = product_weight * x

        assert sp.simplify(out_a - expected.a) == 0
        assert sp.simplify(out_b - expected.b) == 0
