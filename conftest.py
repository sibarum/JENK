"""Pytest config: add project root to sys.path so `import jenk` works.

This avoids needing a pyproject.toml + editable install during early
development. If/when the project grows enough to want proper packaging,
add a pyproject.toml and remove this file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
