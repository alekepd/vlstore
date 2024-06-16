"""Abstractions for serialization."""
from typing import TypeVar, Protocol, Generic, overload, Union, Literal, Optional
from dataclasses import dataclass
from ..store._types import (
    TYPE_RETURNDATA,
    TYPE_INPUTDATA,
    TYPE_OUT,
    T_OUT,
)

_T = TypeVar("_T")
_T_contra = TypeVar("_T_contra", contravariant=True)
_T_co = TypeVar("_T_co", covariant=True)


class Packer(Protocol[_T_contra]):
    """Pack input into binary form."""

    @overload
    def __call__(
        self,
        target: _T_contra,
        /,
        *,
        out: Literal[None] = ...,
    ) -> TYPE_INPUTDATA:
        ...

    @overload
    def __call__(
        self,
        target: _T_contra,
        /,
        *,
        out: T_OUT,
    ) -> T_OUT:
        ...

    def __call__(
        self, target: _T_contra, /, *, out: Optional[TYPE_OUT] = None
    ) -> Union[TYPE_INPUTDATA, TYPE_OUT]:
        """Pack input object.

        Arguments:
        ---------
        target:
            Instance to be packed.
        out:
            If provided, packed bits are places in this object. Must support
            memoryview.

        """


class Unpacker(Protocol[_T_co]):
    """Pack input into binary form."""

    def __call__(self, data: TYPE_RETURNDATA, /) -> _T_co:
        """Unpack bytes into object."""


@dataclass(frozen=True)
class Codec(Generic[_T]):
    """Encode and decode an object to bytes."""

    # todo: should be clear about views/copy

    pack: Packer[_T]
    unpack: Unpacker[_T]
