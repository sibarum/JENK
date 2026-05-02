"""SymbolicNeuron — generic exact-symbolic neuron with built-in update.

The learned parameter is a single sympy expression ``expr`` in a
designated input variable ``var``.  Feedforward substitutes a value for
``var``.  The update rule is the literal residual

    expr  <-  expr + lr * (actual - expected)

with every quantity kept in exact symbolic form (no autodiff, no
gradient over the loss surface, no numeric truncation).  ``actual`` and
``expected`` may themselves be symbolic; ``lr`` may be a sympy Rational,
a free symbol, or any other sympy expression.

Efficiency is not a goal: each step rebuilds the full sympy tree.
"""

from __future__ import annotations

import sympy as sp


class SymbolicNeuron:
    """Generic neuron whose learned parameter is an algebraic expression.

    Fields
    ------
    expr : sympy expression — the learned parameter (mutable).
    var  : sympy symbol — designated input variable inside ``expr``.
           Defaults to ``sp.Symbol('x')``.
    """

    def __init__(self, expr, var=None):
        self.expr = sp.sympify(expr)
        self.var = var if var is not None else sp.Symbol('x')

    def feedforward(self, input_value) -> sp.Expr:
        """Symbolic substitution: ``expr.subs(var, input_value)``.

        ``input_value`` may be a number, a sympy expression, or a Traction
        atom; it is sympified before substitution.
        """
        return self.expr.subs(self.var, sp.sympify(input_value))

    def update(self, actual, expected, lr) -> sp.Expr:
        """Apply  expr <- expr + lr * (actual - expected).

        Returns the delta that was added.  Both ``actual`` and ``expected``
        are kept symbolic; their difference is the loss.
        """
        loss = sp.sympify(actual) - sp.sympify(expected)
        delta = sp.sympify(lr) * loss
        self.expr = self.expr + delta
        return delta

    def step(self, input_value, expected, lr) -> sp.Expr:
        """Feedforward + update in one call.  Returns the forward output.

        This is the all-in-one entry point: substitute the input, compute
        the symbolic residual against ``expected``, scale by ``lr``, and
        add the result back into ``expr``.
        """
        actual = self.feedforward(input_value)
        self.update(actual, expected, lr)
        return actual

    def __call__(self, input_value) -> sp.Expr:
        return self.feedforward(input_value)

    def __repr__(self) -> str:
        return f'SymbolicNeuron(expr={self.expr}, var={self.var})'


class GradientNeuron:
    """Exact-symbolic neuron with proper gradient descent.

    Where SymbolicNeuron treats the whole expression as the thing being
    learned (and the additive rule  expr <- expr + lr*(actual-expected)
    only ever shifts a constant), GradientNeuron treats ``expr`` as a
    *form* whose free symbols other than ``var`` are trainable
    parameters.  Each parameter has a current value in ``params``.

    Loss
    ----
        L = (actual - expected) ** 2

    The bare difference  actual - expected  has no minimum (its gradient
    keeps pushing in one direction forever), so we square it.  Squared
    error is ``>= 0`` with minimum 0 exactly at  actual == expected.

    Update
    ------
    For each trainable symbol ``p``,

        p_value  <-  p_value  -  lr * (dL/dp evaluated at current params)

    Gradients are computed symbolically with ``sp.diff``.  Param values
    can be Rationals (exact, but denominators grow ~1/lr per step) or
    Floats (cheap and what most demos want).

    Construction
    ------------
        x, w, b = sp.symbols('x w b')
        n = GradientNeuron(expr=w*x + b, var=x,
                           params={w: 0, b: 0})
        n.step(input_value=3, expected=11, lr=0.01)
    """

    def __init__(self, expr, var=None, params=None):
        self.expr = sp.sympify(expr)
        self.var = var if var is not None else sp.Symbol('x')
        if params is None:
            params = {p: sp.Integer(0)
                      for p in self.expr.free_symbols if p != self.var}
        self.params = {p: sp.sympify(v) for p, v in params.items()}

    def feedforward(self, input_value) -> sp.Expr:
        """Substitute ``input_value`` for ``var`` and current values for params."""
        partial = self.expr.subs(self.var, sp.sympify(input_value))
        return sp.sympify(partial.subs(self.params))

    def gradients(self, input_value, expected) -> dict:
        """Symbolic dL/dp for each trainable symbol, evaluated at current params.

        L = (expr.subs(var, input_value) - expected) ** 2,  with the params
        kept symbolic for the differentiation; the resulting dL/dp is then
        substituted with the current parameter values.
        """
        actual_sym = self.expr.subs(self.var, sp.sympify(input_value))
        loss = (actual_sym - sp.sympify(expected)) ** 2
        return {p: sp.sympify(sp.diff(loss, p).subs(self.params))
                for p in self.params}

    def update(self, gradients, lr) -> None:
        """Apply  p_value <- p_value - lr * grad  for each trainable symbol."""
        lr_s = sp.sympify(lr)
        for p, g in gradients.items():
            self.params[p] = sp.sympify(self.params[p] - lr_s * g)

    def step(self, input_value, expected, lr) -> sp.Expr:
        """Feedforward + gradient + update in one call.  Returns forward output."""
        actual = self.feedforward(input_value)
        grads = self.gradients(input_value, expected)
        self.update(grads, lr)
        return actual

    def __call__(self, input_value) -> sp.Expr:
        return self.feedforward(input_value)

    def __repr__(self) -> str:
        return (f'GradientNeuron(expr={self.expr}, var={self.var}, '
                f'params={self.params})')
