"""Imports for optional dependencies."""

try:
    import numpy  # noqa: ICN001 F401

    has_numpy = True
except ImportError:
    has_numpy = False


try:
    import flatbuffers  # type: ignore # noqa: F401

    has_flatbuffers = True
except ImportError:
    has_flatbuffers = False
