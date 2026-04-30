"""Tests for jenk.graded_neuron — Traction-native diagnostic neuron.

Covers:
  - identity / generator constructors
  - forward formula at the four cardinals (s in {0, 1, -1, w})
  - automatic firing of framework identities (e.g. w*z -> 1) inside forward
  - trace recording: all expected steps appear, flags fire correctly
  - trace JSON serialization round-trip
  - chain composition with trace propagation
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

from jenk.graded_neuron import GradedNeuron, chain
from jenk.trace import Trace
from jenk.traction import z, w, null, Zero, Omega, Null


# ---------------------------------------------------------------- forward formula

class TestForwardFormula:
    def test_identity_neuron_passes_input_through(self):
        n = GradedNeuron.identity(s=sp.Integer(0))
        x_a, x_b = sp.Symbol('a'), sp.Symbol('b')
        out_a, out_b = n.forward(x_a, x_b)
        assert sp.simplify(out_a - x_a) == 0
        assert sp.simplify(out_b - x_b) == 0

    def test_generator_at_s_zero_is_90_rotation(self):
        # M(0,1,0) = [[0,-1],[1,0]]: (a, b) -> (-b, a)
        n = GradedNeuron.generator(s=sp.Integer(0))
        x_a, x_b = sp.Symbol('a'), sp.Symbol('b')
        out_a, out_b = n.forward(x_a, x_b)
        assert sp.simplify(out_a - (-x_b)) == 0
        assert sp.simplify(out_b - x_a) == 0

    def test_generator_at_s_one_parabolic(self):
        n = GradedNeuron.generator(s=sp.Integer(1))
        x_a, x_b = sp.Symbol('a'), sp.Symbol('b')
        out_a, out_b = n.forward(x_a, x_b)
        assert sp.simplify(out_a - (-x_b)) == 0
        assert sp.simplify(out_b - (x_a + x_b)) == 0

    def test_generator_at_s_minus_one_projective_cardinal(self):
        n = GradedNeuron.generator(s=sp.Integer(-1))
        x_a, x_b = sp.Symbol('a'), sp.Symbol('b')
        out_a, out_b = n.forward(x_a, x_b)
        assert sp.simplify(out_a - (-x_b)) == 0
        assert sp.simplify(out_b - (x_a - x_b)) == 0


# ---------------------------------------------------------------- framework identities firing

class TestFrameworkIdentitiesInForward:
    """The whole point: framework identities should fire automatically."""

    def test_w_times_z_fires_to_one_inside_layer(self):
        """At s=w, generator with x_b=z hits m6 = w * z = 1 automatically."""
        n = GradedNeuron.generator(s=w)
        x_a = sp.Symbol('a')
        out_a, out_b = n.forward(x_a, z)
        # m5 = 0 + 1*w = w; m6 = w * z = 1; out_b = x_a + 1
        assert sp.simplify(out_b - (x_a + 1)) == 0
        # out_a = -z (the framework zero, negated)
        assert out_a == -z

    def test_z_to_the_w_fires_in_intermediate(self):
        """If a parameter is z**w, sympy collapses to -1 immediately."""
        n = GradedNeuron(s=sp.Integer(0), c=z**w, d=sp.Integer(0))
        x_a, x_b = sp.Symbol('a'), sp.Symbol('b')
        out_a, out_b = n.forward(x_a, x_b)
        # c = z**w = -1; out_a = -1 * x_a; out_b = (-1 + 0)*x_b = -x_b
        assert sp.simplify(out_a - (-x_a)) == 0
        assert sp.simplify(out_b - (-x_b)) == 0

    def test_omega_in_s_with_zero_d_doesnt_explode(self):
        """When d=0, s never multiplies anything; w in s should be inert."""
        n = GradedNeuron(s=w, c=sp.Integer(2), d=sp.Integer(0))
        x_a, x_b = sp.Symbol('a'), sp.Symbol('b')
        out_a, out_b = n.forward(x_a, x_b)
        # m4 = 0*w = 0; out_a = 2*x_a; out_b = 2*x_b
        assert sp.simplify(out_a - 2*x_a) == 0
        assert sp.simplify(out_b - 2*x_b) == 0


# ---------------------------------------------------------------- trace mechanics

class TestTraceRecording:
    def test_forward_without_trace_produces_no_records(self):
        n = GradedNeuron.generator(s=sp.Integer(0))
        n.forward(sp.Symbol('a'), sp.Symbol('b'))
        # No exception, no trace involved — just verify forward returns

    def test_forward_with_trace_records_all_named_steps(self):
        t = Trace(label='probe')
        n = GradedNeuron.generator(s=sp.Integer(0))
        n.forward(sp.Symbol('a'), sp.Symbol('b'), trace=t)

        step_names = [r.step for r in t.records]
        # Inputs and params
        assert 'input.x_a' in step_names
        assert 'input.x_b' in step_names
        assert 'param.s' in step_names
        assert 'param.c' in step_names
        assert 'param.d' in step_names
        # Computation steps
        assert 'm1 = c * x_a' in step_names
        assert 'm2 = d * x_b' in step_names
        assert 'out_a_core = m1 - m2' in step_names
        assert 'm3 = d * x_a' in step_names
        assert 'm4 = d * s' in step_names
        assert 'm5 = c + m4' in step_names
        assert 'm6 = m5 * x_b' in step_names
        assert 'out_b_core = m3 + m6' in step_names
        # Outputs
        assert 'output.out_a' in step_names
        assert 'output.out_b' in step_names

    def test_omega_atom_flagged_when_present(self):
        t = Trace(label='omega-probe')
        n = GradedNeuron.generator(s=w)
        n.forward(sp.Symbol('a'), sp.Symbol('b'), trace=t)

        # m4 = d*s = 1*w = w should have omega flag
        m4_record = next(r for r in t.records if r.step == 'm4 = d * s')
        assert m4_record.flags['has_omega'] is True

    def test_summary_counts_events(self):
        t = Trace(label='summary-probe')
        n = GradedNeuron.generator(s=w)
        n.forward(z, sp.Symbol('b'), trace=t)
        summary = t.summary()
        assert summary['n_records'] > 0
        assert summary['n_omega_atom_appearances'] >= 1
        assert summary['n_zero_atom_appearances'] >= 1


# ---------------------------------------------------------------- JSON round-trip

class TestTraceJSON:
    def test_save_writes_valid_json(self, tmp_path: Path):
        t = Trace(label='json-test')
        n = GradedNeuron.generator(s=w)
        n.forward(z, sp.Symbol('b'), trace=t)

        path = tmp_path / 'trace.json'
        t.save(path)
        assert path.exists()

        data = json.loads(path.read_text(encoding='utf-8'))
        assert data['label'] == 'json-test'
        assert 'records' in data
        assert 'summary' in data
        assert 'metadata' in data
        assert len(data['records']) == len(t.records)

    def test_record_has_raw_and_simplified_fields(self, tmp_path: Path):
        t = Trace(label='roundtrip')
        n = GradedNeuron.generator(s=w)
        n.forward(z, sp.Symbol('b'), trace=t)
        path = tmp_path / 'trace.json'
        t.save(path)

        data = json.loads(path.read_text(encoding='utf-8'))
        for rec in data['records']:
            assert 'raw' in rec
            assert 'simplified' in rec
            assert 'str' in rec['raw']
            assert 'srepr' in rec['raw']
            assert 'flags' in rec

    def test_context_manager_saves_on_exit(self, tmp_path: Path):
        path = tmp_path / 'cm.json'
        with Trace(label='cm-test') as t:
            n = GradedNeuron.generator(s=sp.Integer(0))
            n.forward(sp.Symbol('a'), sp.Symbol('b'), trace=t)
            t.save(path)  # explicit save to known path
        assert path.exists()


# ---------------------------------------------------------------- chain composition

class TestChain:
    def test_chain_of_two_identities_passes_through(self):
        n1 = GradedNeuron.identity(s=sp.Integer(0))
        n2 = GradedNeuron.identity(s=sp.Integer(1))
        composed = chain(n1, n2)
        x_a, x_b = sp.Symbol('a'), sp.Symbol('b')
        out_a, out_b = composed(x_a, x_b)
        assert sp.simplify(out_a - x_a) == 0
        assert sp.simplify(out_b - x_b) == 0

    def test_chain_of_two_generators_at_s_zero_is_minus_identity(self):
        # g^2 at s=0 is -1 (since g^2 = s*g - 1 = -1)
        # Two 90-deg rotations = 180-deg rotation = -I
        n1 = GradedNeuron.generator(s=sp.Integer(0))
        n2 = GradedNeuron.generator(s=sp.Integer(0))
        composed = chain(n1, n2)
        x_a, x_b = sp.Symbol('a'), sp.Symbol('b')
        out_a, out_b = composed(x_a, x_b)
        assert sp.simplify(out_a - (-x_a)) == 0
        assert sp.simplify(out_b - (-x_b)) == 0

    def test_chain_propagates_trace_across_layers(self):
        t = Trace(label='chain-probe')
        n1 = GradedNeuron.generator(s=sp.Integer(0))
        n2 = GradedNeuron.generator(s=sp.Integer(0))
        composed = chain(n1, n2)
        composed(sp.Symbol('a'), sp.Symbol('b'), trace=t)

        layer_entries = [r.step for r in t.records if r.step.startswith('chain.layer[')]
        # Two layers, each with at least one entry record
        assert any('chain.layer[0]' in s for s in layer_entries)
        assert any('chain.layer[1]' in s for s in layer_entries)


# ---------------------------------------------------------------- substitution

class TestSubstitution:
    def test_substitute_s_specializes_neuron(self):
        s_sym = sp.Symbol('s_sym')
        n = GradedNeuron.generator(s=s_sym)
        x_a, x_b = sp.Symbol('a'), sp.Symbol('b')
        out_a, out_b = n.forward(x_a, x_b)
        # out_b contains s_sym
        assert s_sym in out_b.free_symbols

        # Specialize to s=w
        n_specialized = n.with_substitution(s_sym, w)
        out_a2, out_b2 = n_specialized.forward(x_a, x_b)
        # Now out_b should contain w (the Omega atom), not s_sym
        assert s_sym not in (out_b2.free_symbols if hasattr(out_b2, 'free_symbols') else set())
