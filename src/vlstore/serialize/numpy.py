"""Serialization tools for numpy arrays."""
from typing import overload, Union, Literal, Optional
import numpy as np
from ..store._types import (
    TYPE_RETURNDATA,
    TYPE_OUT,
    T_OUT,
)
from ._types import Codec
from ..store.util import bytewise_memoryview


@overload
def _checked_numpy_f32_packer(
    x: np.ndarray,
    /,
    *,
    out: Literal[None] = ...,
) -> memoryview:
    ...


@overload
def _checked_numpy_f32_packer(
    x: np.ndarray,
    /,
    *,
    out: T_OUT,
) -> T_OUT:
    ...


def _checked_numpy_f32_packer(
    x: np.ndarray, *, out: Optional[TYPE_OUT] = None
) -> Union[memoryview, TYPE_OUT]:
    if not x.flags["C_CONTIGUOUS"]:
        raise ValueError("Only C-contiguous arrays may be serialized.")
    if x.dtype is not np.dtype('float32'):
        raise ValueError("Data must be float32.")
    data = bytewise_memoryview(x)
    if out is None:
        return data
    else:
        view = bytewise_memoryview(out)
        view[:] = data
        return view


def _numpy_f32_unpacker(
    x: TYPE_RETURNDATA,
) -> np.ndarray:
    view = bytewise_memoryview(x)
    return np.frombuffer(view,dtype=np.dtype('float32'))


flatfloatndarray_codec = Codec[np.ndarray](
    pack=_checked_numpy_f32_packer, unpack=_numpy_f32_unpacker
)
