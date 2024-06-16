"""Imports for optional dependencies."""

try:
    import numpy  # noqa

    has_numpy = True
except ImportError:
    has_numpy = False
