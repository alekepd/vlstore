"""Contains type definitions for the store submodule.

Some of these get complex in order to deal with optional dependencies.
"""
from typing import Union, TypeVar, Protocol, TYPE_CHECKING
from typing_extensions import TypeAlias
from .._optional_dependencies import has_numpy


class SupportsBuffer(Protocol):
    """Stand in for object that supports the buffer interface.

    This likely will not catch all buffer-compatible types.

    Requires ideas in PEP688, which are too new for the required python version.

    """

    def __buffer__(self, flags: int) -> memoryview:
        """Call when buffer is requested."""

    def __releate_buffer__(self, buffer: memoryview) -> None:
        """Call when buffer is no longer needed."""


# we only want to include certain type definitions if numpy is installed.
# the following approach really won't scale well if we need to do this for
# more libraries.

# intended purpose: if not type checking or with mypy, code should execute.
if TYPE_CHECKING or has_numpy:
    import numpy as np

    # we should get numpy out of this file
    TYPE_OUT: TypeAlias = Union[bytearray, memoryview, SupportsBuffer, np.ndarray]
    TYPE_RETURNDATA: TypeAlias = Union[
        bytes, bytearray, memoryview, SupportsBuffer, np.ndarray
    ]
    TYPE_INPUTDATA: TypeAlias = Union[bytes, bytearray, memoryview, np.ndarray]
else:
    TYPE_OUT = Union[bytearray, memoryview, SupportsBuffer]
    TYPE_RETURNDATA = Union[bytes, bytearray, memoryview, SupportsBuffer]
    TYPE_INPUTDATA = Union[bytes, bytearray, memoryview]

TYPE_MINIMAL_RETURNDATA = Union[bytes, bytearray, memoryview]
TYPE_KEY = Union[int, str, bytes]

T_OUT = TypeVar("T_OUT", bound=TYPE_OUT)
