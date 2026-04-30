"""Diagnostic trace recorder for symbolic forward passes.

A `Trace` accumulates `Record`s as a computation runs. Each record captures
the raw sympy expression at a step plus its `traction_simplify`-reduced
form, side by side, so you can see where framework identities fired and
where erasure events (Null) showed up.

Usage:
    from jenk.trace import Trace
    from jenk.graded_neuron import GradedNeuron

    with Trace(label='four cardinals probe') as t:
        n = GradedNeuron.generator(s=w)
        n.forward(x_a, x_b, trace=t)
    # writes tmp/diagnostics/<timestamp>.json on exit

The trace can also be saved manually via `t.save(path)`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import sympy as sp

from jenk.traction import Null, Zero, Omega, traction_simplify


def _serialize_expr(expr: sp.Expr) -> dict:
    """Serialize a sympy expression as both human-readable and round-trippable."""
    return {
        'str': str(expr),
        'srepr': sp.srepr(expr),
    }


def _has_null(expr: sp.Expr) -> bool:
    """True if `null` (erasure) appears anywhere in the expression."""
    return bool(expr.has(Null)) if hasattr(expr, 'has') else False


def _has_zero_atom(expr: sp.Expr) -> bool:
    """True if the framework Zero atom appears anywhere in the expression."""
    return bool(expr.has(Zero)) if hasattr(expr, 'has') else False


def _has_omega_atom(expr: sp.Expr) -> bool:
    """True if the framework Omega atom appears anywhere in the expression."""
    return bool(expr.has(Omega)) if hasattr(expr, 'has') else False


@dataclass
class Record:
    step: str
    raw: dict
    simplified: dict
    operands: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    note: str = ''

    @classmethod
    def from_value(
        cls,
        step: str,
        value: sp.Expr,
        operands: dict[str, sp.Expr] | None = None,
        note: str = '',
    ) -> 'Record':
        value = sp.sympify(value)
        simplified = traction_simplify(value)
        return cls(
            step=step,
            raw=_serialize_expr(value),
            simplified=_serialize_expr(simplified),
            operands={k: _serialize_expr(v) for k, v in (operands or {}).items()},
            flags={
                'has_null': _has_null(simplified),
                'has_zero': _has_zero_atom(simplified),
                'has_omega': _has_omega_atom(simplified),
                'differs_from_raw': str(value) != str(simplified),
            },
            note=note,
        )

    def to_dict(self) -> dict:
        return {
            'step': self.step,
            'raw': self.raw,
            'simplified': self.simplified,
            'operands': self.operands,
            'flags': self.flags,
            'note': self.note,
        }


@dataclass
class Trace:
    label: str = ''
    metadata: dict = field(default_factory=dict)
    records: list[Record] = field(default_factory=list)
    path: Path | None = None

    def __post_init__(self) -> None:
        self.metadata.setdefault('created_at', datetime.now().isoformat(timespec='seconds'))

    def record(
        self,
        step: str,
        value: sp.Expr,
        operands: dict[str, sp.Expr] | None = None,
        note: str = '',
    ) -> sp.Expr:
        """Record one step. Returns `value` unchanged so callers can chain."""
        self.records.append(Record.from_value(step, value, operands=operands, note=note))
        return value

    def to_dict(self) -> dict:
        return {
            'label': self.label,
            'metadata': self.metadata,
            'records': [r.to_dict() for r in self.records],
            'summary': self.summary(),
        }

    def summary(self) -> dict:
        """Quick-glance counts of interesting events across all records."""
        return {
            'n_records': len(self.records),
            'n_null_events': sum(1 for r in self.records if r.flags.get('has_null')),
            'n_zero_atom_appearances': sum(1 for r in self.records if r.flags.get('has_zero')),
            'n_omega_atom_appearances': sum(1 for r in self.records if r.flags.get('has_omega')),
            'n_simplifications': sum(1 for r in self.records if r.flags.get('differs_from_raw')),
        }

    def save(self, path: Path | str | None = None) -> Path:
        """Write the trace as JSON. Default path: tmp/diagnostics/<timestamp>_<label>.json."""
        if path is None:
            path = self._default_path()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        self.path = path
        return path

    def _default_path(self) -> Path:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        slug = ''.join(c if c.isalnum() or c in '-_' else '_' for c in self.label) or 'trace'
        return Path('tmp/diagnostics') / f'{ts}_{slug}.json'

    def __enter__(self) -> 'Trace':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Always save on exit, even on exception, so failures are diagnosable.
        try:
            self.save()
        except Exception:
            # Don't shadow original exceptions during traceback
            pass
