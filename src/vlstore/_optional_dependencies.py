# ruff: noqa
"""Imports for optional dependencies."""

try:
    import numpy

    has_numpy = True
except ImportError:
    has_numpy = False


try:
    import flatbuffers # type: ignore

    has_flatbuffers = True
except ImportError:
    has_flatbuffers = False
