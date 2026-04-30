"""LearnableNeuron — symbolic-first gradient-trainable layer.

This is the Stage-5 entry point: a neuron whose `s, c, d, bias_a, bias_b`
can be declared as *learnable parameters* and updated by gradient descent,
with the entire forward pass and gradient remaining symbolic. Floats only
appear at two narrow edges: the current numeric value of each parameter,
and the final scalar gradient extracted at a substitution. The framework
atoms (Zero, Omega, Null) are never coerced.

Why symbolic-first
------------------

Past attempts at embedding Traction semantics in a numerical layer
(PyTorch / numpy float-from-the-start) ran into "lost in translation"
pathologies: framework identities that fire automatically in sympy
(`0·ω → 1`, `0^ω → -1`, etc.) silently disappear when atoms are coerced
to floats. The reference implementation therefore stays sympy-throughout,
and the diagnostic Trace can record the gradient expression itself at
every step — gradient descent is fully inspectable.

This is slow at scale; that is acceptable. The reference is for
correctness-first; numerical performance is a separate later stage that
must be validated against this reference.

Design
------

A `LearnableParameter` is a sympy `Symbol` paired with a current numeric
value. Anywhere a `LearnableParameter` is passed where the underlying
neuron expects a sympy `Expr`, its `.symbol` is used in the symbolic
forward pass and its `.value` is used for substitution at evaluation
time.

A `LearnableNeuron` wraps `GradedNeuron`. Each construction parameter
can be either a fixed sympy expression (held constant) or a
`LearnableParameter` (tracked for training). The forward pass returns a
symbolic expression; `evaluate(expr)` substitutes current values.

A `SymbolicOptimizer` takes a `LearnableNeuron` and a symbolic loss
expression and runs gradient descent. It builds the symbolic gradient
once via `sp.diff`, lambdifies it for fast repeated numeric evaluation,
and updates each parameter's `.value` in place.

Atoms in fixed parameters (e.g. `c=0`, `d=z`, `s=w`) are supported —
those parameters simply carry the framework atoms through the symbolic
forward pass as usual. Atoms in *learnable* parameters are not currently
supported; the M_proj findings established that real-valued `s` covers
three of four cardinals (hyp at 0, par at 1, ell at -1), and ω is a
discrete substitution rather than a smooth limit. Reaching ω through
training is a separate later question and connects to James's spike
hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import sympy as sp

from jenk.graded_neuron import GradedNeuron, Pair
from jenk.trace import Trace


@dataclass
class LearnableParameter:
    """A sympy Symbol paired with its current numeric value.

    The `.symbol` is what appears in the symbolic forward pass; the
    `.value` is what gets substituted in for evaluation and is what
    gradient descent updates.
    """
    symbol: sp.Symbol
    value: float

    @staticmethod
    def named(name: str, initial: float, *, real: bool = True) -> 'LearnableParameter':
        """Convenience: create a real-valued learnable parameter from a name string."""
        return LearnableParameter(sp.Symbol(name, real=real), float(initial))


def _coerce(p):
    """Pull the symbolic representation out of a parameter (learnable or fixed)."""
    if isinstance(p, LearnableParameter):
        return p.symbol
    return sp.sympify(p)


@dataclass
class LearnableNeuron:
    """A GradedNeuron whose parameters can be marked learnable.

    Construction takes the same five fields as GradedNeuron, but each may
    be either a fixed sympy expression (any sympy `Expr`, including
    framework atoms) or a `LearnableParameter`. The forward pass uses the
    parameter's symbol; `evaluate(expr)` substitutes the parameter's
    current value.
    """
    s: object
    c: object
    d: object
    bias_a: object = 0
    bias_b: object = 0

    _learnable: list[LearnableParameter] = field(default_factory=list, init=False, repr=False)
    _neuron: GradedNeuron = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for slot in ('s', 'c', 'd', 'bias_a', 'bias_b'):
            value = getattr(self, slot)
            if isinstance(value, LearnableParameter):
                self._learnable.append(value)
        self._neuron = GradedNeuron(
            s=_coerce(self.s),
            c=_coerce(self.c),
            d=_coerce(self.d),
            bias_a=_coerce(self.bias_a),
            bias_b=_coerce(self.bias_b),
        )

    # --- Forward ---------------------------------------------------------

    def forward(self, x_a, x_b, trace: Trace | None = None) -> Pair:
        """Symbolic forward pass — returns sympy expressions in the parameter symbols."""
        return self._neuron.forward(x_a, x_b, trace=trace)

    def __call__(self, x_a, x_b, trace: Trace | None = None) -> Pair:
        return self.forward(x_a, x_b, trace=trace)

    # --- Parameter introspection ----------------------------------------

    def parameters(self) -> list[LearnableParameter]:
        """Return the list of learnable parameters in declaration order."""
        return list(self._learnable)

    def parameter_symbols(self) -> list[sp.Symbol]:
        """Return just the sympy Symbols of the learnable parameters."""
        return [p.symbol for p in self._learnable]

    def substitution_dict(self) -> dict:
        """Return {symbol: current_value} for use with sp.subs."""
        return {p.symbol: sp.Float(p.value) for p in self._learnable}

    def evaluate(self, expr: sp.Expr) -> sp.Expr:
        """Substitute current parameter values into a symbolic expression.

        Returns a sympy expression. If `expr` was numeric in the parameters
        and contains no framework atoms, the result will be a sympy Float.
        If it contains atoms (Zero, Omega, Null), the result still contains
        them — call `project_complex` separately to land in ℂ.
        """
        return expr.subs(self.substitution_dict())

    def evaluate_float(self, expr: sp.Expr) -> float:
        """Substitute and convert to a Python float. Raises if not a number."""
        result = self.evaluate(expr)
        return float(result)


@dataclass
class SymbolicOptimizer:
    """Symbolic gradient descent over a LearnableNeuron's parameters.

    Build the loss as a sympy expression (typically squared error between
    a `LearnableNeuron.forward()` output and a target). Pass the same
    expression to `step()` repeatedly; the optimizer caches the
    lambdified gradient so repeated steps are fast.

    Symbolic-first contract: `loss_expr` may contain framework atoms in
    fixed parts. The atoms are projected via `project_complex` only at
    the moment a numeric loss / gradient value is needed. Atom integrity
    is preserved up to that boundary.
    """
    neuron: LearnableNeuron
    learning_rate: float = 0.1

    _grad_lambdas: dict[sp.Symbol, Callable] = field(default_factory=dict, init=False, repr=False)
    _loss_lambda: Callable | None = field(default=None, init=False, repr=False)
    _cached_loss: sp.Expr | None = field(default=None, init=False, repr=False)

    def _ensure_compiled(self, loss_expr: sp.Expr) -> None:
        """Build (or reuse) lambdified loss + gradient callables."""
        if self._cached_loss is loss_expr:
            return
        # Project atoms in the loss expression to ℂ before lambdify;
        # otherwise sympy can't compile a numeric callable.
        from jenk.traction import project_complex
        projected = project_complex(loss_expr)
        symbols = self.neuron.parameter_symbols()
        self._loss_lambda = sp.lambdify(symbols, projected, modules=['numpy'])
        self._grad_lambdas = {}
        for sym in symbols:
            grad_expr = sp.diff(loss_expr, sym)
            grad_projected = project_complex(grad_expr)
            self._grad_lambdas[sym] = sp.lambdify(symbols, grad_projected, modules=['numpy'])
        self._cached_loss = loss_expr

    def loss_value(self, loss_expr: sp.Expr) -> float:
        """Numeric loss at the current parameter values."""
        self._ensure_compiled(loss_expr)
        values = [p.value for p in self.neuron.parameters()]
        result = self._loss_lambda(*values)
        # The complex projection may give a real result with imaginary 0; coerce.
        if hasattr(result, 'real') and abs(getattr(result, 'imag', 0.0)) < 1e-12:
            return float(result.real)
        return float(result)

    def gradient_expr(self, loss_expr: sp.Expr, param: LearnableParameter) -> sp.Expr:
        """Return the symbolic gradient ∂loss/∂param.symbol — useful for tracing."""
        return sp.diff(loss_expr, param.symbol)

    def gradient_value(self, loss_expr: sp.Expr, param: LearnableParameter) -> float:
        """Numeric gradient ∂loss/∂param at the current parameter values."""
        self._ensure_compiled(loss_expr)
        values = [p.value for p in self.neuron.parameters()]
        result = self._grad_lambdas[param.symbol](*values)
        if hasattr(result, 'real') and abs(getattr(result, 'imag', 0.0)) < 1e-12:
            return float(result.real)
        return float(result)

    def step(self, loss_expr: sp.Expr) -> float:
        """One gradient-descent step. Returns the loss value *before* the step."""
        self._ensure_compiled(loss_expr)
        loss_before = self.loss_value(loss_expr)
        # Compute all gradients at the current point before any update,
        # so the step is a true joint update.
        grads = {
            p.symbol: self.gradient_value(loss_expr, p)
            for p in self.neuron.parameters()
        }
        for p in self.neuron.parameters():
            p.value -= self.learning_rate * grads[p.symbol]
        return loss_before


def squared_error_loss(
    neuron: LearnableNeuron,
    pairs: list[tuple[tuple, tuple]],
) -> sp.Expr:
    """Build a sympy squared-error loss over (input, target) pairs.

    `pairs` is a list of ((x_a, x_b), (target_a, target_b)) tuples where each
    component is anything sympy can sympify (Python number, sympy Expr, atom).

    Returns a single sympy expression that depends on the neuron's learnable
    parameter symbols (and, possibly, framework atoms in fixed parameters).
    """
    total = sp.Integer(0)
    for (x_a, x_b), (target_a, target_b) in pairs:
        out_a, out_b = neuron.forward(x_a, x_b)
        total = total + (out_a - sp.sympify(target_a)) ** 2
        total = total + (out_b - sp.sympify(target_b)) ** 2
    return total
