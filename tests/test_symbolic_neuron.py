"""Tests for jenk.symbolic_neuron — the symbolic-prototype neuron.

Three banks of probes:

  (1) Forward formula      — output matches the regular-rep matrix product,
                              biases add correctly, linearity in input.
  (2) Cardinal-mode probe  — substitute s in {0, 1, omega, -1} and
                              verify each cardinal output explicitly.
                              Cross-check carrier_g_squared values.
  (3) Composition          — chained neurons equal the matrix product;
                              omega survives a mixed chain symbolically;
                              identity neuron is the identity.
"""

import pytest
import sympy as sp

from jenk.graded import Graded, omega
from jenk.symbolic_neuron import SymbolicNeuron, chain


# Free symbols used as inputs and parameters.
x_a, x_b = sp.symbols('x_a x_b', real=True)
c, d = sp.symbols('c d', real=True)
s_sym = sp.Symbol('s_sym', real=True)


# --- Helpers --------------------------------------------------------------

def expr_equal(a, b) -> bool:
    return sp.simplify(a - b) == 0


def pair_equal(p1, p2) -> bool:
    return expr_equal(p1[0], p2[0]) and expr_equal(p1[1], p2[1])


# --- Bank 1: forward formula ---------------------------------------------

class TestForwardFormula:
    def test_matches_regular_rep_matrix(self):
        n = SymbolicNeuron(s=s_sym, c=c, d=d)
        out_a, out_b = n.forward(x_a, x_b)
        # M(c, d, s) acting on (x_a, x_b)^T
        M = sp.Matrix([[c, -d], [d, c + d * s_sym]])
        v = sp.Matrix([x_a, x_b])
        expected = M * v
        assert expr_equal(out_a, expected[0])
        assert expr_equal(out_b, expected[1])

    def test_matrix_method_returns_correct_matrix(self):
        n = SymbolicNeuron(s=s_sym, c=c, d=d)
        M = n.matrix()
        assert expr_equal(M[0, 0], c)
        assert expr_equal(M[0, 1], -d)
        assert expr_equal(M[1, 0], d)
        assert expr_equal(M[1, 1], c + d * s_sym)

    def test_bias_added_to_output(self):
        ba, bb = sp.symbols('ba bb', real=True)
        n_bias = SymbolicNeuron(s=s_sym, c=c, d=d, bias_a=ba, bias_b=bb)
        n_nobias = SymbolicNeuron(s=s_sym, c=c, d=d)
        out_with = n_bias.forward(x_a, x_b)
        out_no = n_nobias.forward(x_a, x_b)
        assert expr_equal(out_with[0], out_no[0] + ba)
        assert expr_equal(out_with[1], out_no[1] + bb)

    def test_linearity_in_input_when_bias_zero(self):
        """N(alpha*u + beta*v) = alpha*N(u) + beta*N(v)  when bias=0."""
        n = SymbolicNeuron(s=s_sym, c=c, d=d)
        u_a, u_b, v_a, v_b = sp.symbols('u_a u_b v_a v_b', real=True)
        alpha, beta = sp.symbols('alpha beta', real=True)
        lhs = n.forward(alpha * u_a + beta * v_a,
                        alpha * u_b + beta * v_b)
        nu = n.forward(u_a, u_b)
        nv = n.forward(v_a, v_b)
        rhs = (alpha * nu[0] + beta * nv[0],
               alpha * nu[1] + beta * nv[1])
        assert pair_equal(lhs, rhs)

    def test_call_alias(self):
        n = SymbolicNeuron(s=s_sym, c=c, d=d)
        assert pair_equal(n(x_a, x_b), n.forward(x_a, x_b))

    def test_python_int_inputs_get_sympified(self):
        n = SymbolicNeuron(s=0, c=0, d=1)
        # Generator at s=0: matrix is [[0,-1],[1,0]], so (3, 4) -> (-4, 3).
        out_a, out_b = n.forward(3, 4)
        assert out_a == sp.Integer(-4)
        assert out_b == sp.Integer(3)


# --- Bank 2: cardinal-mode probe -----------------------------------------

class TestCardinalModeProbe:
    """Build one generator-weight neuron with symbolic s, substitute s
    in {0, 1, omega, -1}, and inspect each output explicitly. This is
    the 'is the projective mode actually saying something' test.
    """

    def test_generator_neuron_symbolic_output(self):
        """At symbolic s, the generator neuron computes (-x_b, x_a + s*x_b)."""
        n = SymbolicNeuron.generator(s_sym)
        out = n.forward(x_a, x_b)
        assert expr_equal(out[0], -x_b)
        assert expr_equal(out[1], x_a + s_sym * x_b)

    def test_substitute_s_zero_hyperbolic_cardinal(self):
        """At s=0 (framework hyperbolic mode j): (-x_b, x_a)."""
        n = SymbolicNeuron.generator(s_sym)
        out = n.forward(x_a, x_b)
        out_at_0 = (out[0].subs(s_sym, 0), out[1].subs(s_sym, 0))
        assert pair_equal(out_at_0, (-x_b, x_a))

    def test_substitute_s_one_parabolic_cardinal(self):
        """At s=1 (framework parabolic mode e): (-x_b, x_a + x_b)."""
        n = SymbolicNeuron.generator(s_sym)
        out = n.forward(x_a, x_b)
        out_at_1 = (out[0].subs(s_sym, 1), out[1].subs(s_sym, 1))
        assert pair_equal(out_at_1, (-x_b, x_a + x_b))

    def test_substitute_s_minus_one_projective_cardinal(self):
        """At s=-1 (framework projective mode k): (-x_b, x_a - x_b)."""
        n = SymbolicNeuron.generator(s_sym)
        out = n.forward(x_a, x_b)
        out_at_m1 = (out[0].subs(s_sym, -1), out[1].subs(s_sym, -1))
        assert pair_equal(out_at_m1, (-x_b, x_a - x_b))

    def test_substitute_s_omega_elliptic_cardinal(self):
        """At s=omega (framework elliptic mode n): (-x_b, x_a + omega*x_b).

        Crucially, omega survives in the output as a free symbol.
        Nothing collapses it; nothing replaces it with a numeric proxy.
        Whether this 'holds water' is for downstream analysis to decide.
        """
        n = SymbolicNeuron.generator(s_sym)
        out = n.forward(x_a, x_b)
        out_at_omega = (out[0].subs(s_sym, omega),
                        out[1].subs(s_sym, omega))
        assert pair_equal(out_at_omega, (-x_b, x_a + omega * x_b))
        # And omega really is still there in the second component.
        assert omega in out_at_omega[1].free_symbols

    def test_directly_constructed_omega_neuron_matches_substitution(self):
        """Building a neuron with s=omega directly equals substitute-into-symbolic."""
        n_direct = SymbolicNeuron.generator(omega)
        out_direct = n_direct.forward(x_a, x_b)
        n_sym = SymbolicNeuron.generator(s_sym)
        out_sub = n_sym.forward(x_a, x_b)
        out_sub = (out_sub[0].subs(s_sym, omega),
                   out_sub[1].subs(s_sym, omega))
        assert pair_equal(out_direct, out_sub)

    @pytest.mark.parametrize("s_val,expected", [
        (sp.Integer(0),   Graded(sp.Integer(1),  sp.Integer(0))),    # j: 1
        (sp.Integer(1),   Graded(sp.Integer(1),  sp.Integer(1))),    # e: 0
        (omega,           Graded(sp.Integer(-1), sp.Integer(0))),    # n: -1 via wrap
        (sp.Integer(-1),  Graded(sp.Integer(-1), sp.Integer(1))),    # k: -omega via sign-flip
    ])
    def test_carrier_g_squared_at_cardinals(self, s_val, expected):
        """The carrier-side g(s)^2 reduces to the expected cardinal target.

        This is independent of the matrix; it asks only what the carrier
        thinks g(s)^2 should be at each cardinal s.
        """
        n = SymbolicNeuron.generator(s_val)
        got = n.carrier_g_squared()
        assert sp.simplify(got.a - expected.a) == 0
        assert sp.simplify(got.b - expected.b) == 0


# --- Bank 3: composition --------------------------------------------------

class TestComposition:
    def test_two_neurons_compose_as_matrix_product(self):
        """chain(n1, n2)(v) = (M2 * M1) * v ."""
        s1, s2 = sp.symbols('s1 s2', real=True)
        n1 = SymbolicNeuron.generator(s1)
        n2 = SymbolicNeuron.generator(s2)
        out = chain(n1, n2)(x_a, x_b)
        M1, M2 = n1.matrix(), n2.matrix()
        v = sp.Matrix([x_a, x_b])
        expected = (M2 * M1) * v
        assert expr_equal(out[0], expected[0])
        assert expr_equal(out[1], expected[1])

    def test_chain_with_omega_neuron_keeps_omega(self):
        """Compose a generator neuron at s=omega with one at s=0; omega survives."""
        n_om = SymbolicNeuron.generator(omega)
        n_0 = SymbolicNeuron.generator(sp.Integer(0))
        out = chain(n_om, n_0)(x_a, x_b)
        # n_om: (x_a, x_b) -> (-x_b, x_a + omega*x_b)
        # n_0:  (a, b)     -> (-b, a)
        # composed: (-x_a - omega*x_b, -x_b)
        assert expr_equal(out[0], -x_a - omega * x_b)
        assert expr_equal(out[1], -x_b)
        assert omega in sp.simplify(out[0]).free_symbols

    def test_identity_neuron_is_identity(self):
        """Neuron with c=1, d=0 leaves input unchanged regardless of s."""
        n = SymbolicNeuron.identity(s_sym)
        out = n.forward(x_a, x_b)
        assert pair_equal(out, (x_a, x_b))

    def test_chain_singleton_equals_neuron(self):
        n = SymbolicNeuron.generator(s_sym)
        assert pair_equal(chain(n)(x_a, x_b), n.forward(x_a, x_b))

    def test_chain_three_neurons(self):
        s1, s2, s3 = sp.symbols('s1 s2 s3', real=True)
        n1 = SymbolicNeuron.generator(s1)
        n2 = SymbolicNeuron.generator(s2)
        n3 = SymbolicNeuron.generator(s3)
        out = chain(n1, n2, n3)(x_a, x_b)
        M1, M2, M3 = n1.matrix(), n2.matrix(), n3.matrix()
        v = sp.Matrix([x_a, x_b])
        expected = (M3 * M2 * M1) * v
        assert expr_equal(out[0], expected[0])
        assert expr_equal(out[1], expected[1])


# --- Bank 4: substitution / specialization -------------------------------

class TestSubstitution:
    def test_with_substitution_specializes_s(self):
        n = SymbolicNeuron(s=s_sym, c=c, d=d)
        n_at_0 = n.with_substitution(s_sym, 0)
        out = n_at_0.forward(x_a, x_b)
        # At s=0: out = (c*x_a - d*x_b, d*x_a + c*x_b)
        assert pair_equal(out, (c * x_a - d * x_b, d * x_a + c * x_b))

    def test_with_substitution_substitutes_in_all_fields(self):
        # If a parameter is itself a function of a symbol, substitution
        # should propagate to bias, c, d, etc.
        u = sp.Symbol('u', real=True)
        n = SymbolicNeuron(s=u, c=u + 1, d=u, bias_a=u**2, bias_b=u - 1)
        n_at_2 = n.with_substitution(u, 2)
        assert n_at_2.s == sp.Integer(2)
        assert n_at_2.c == sp.Integer(3)
        assert n_at_2.d == sp.Integer(2)
        assert n_at_2.bias_a == sp.Integer(4)
        assert n_at_2.bias_b == sp.Integer(1)
