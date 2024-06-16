"""Utilizes fo rdealin with byte-wise data representations."""
from typing import TYPE_CHECKING, Dict, Any, Callable, Union
from ._types import TYPE_INPUTDATA, TYPE_RETURNDATA
from .._optional_dependencies import has_numpy

_lengthers: Dict[Any, Callable] = {bytes: len, bytearray: len, memoryview: len}
if TYPE_CHECKING or has_numpy:
    import numpy as np

    _lengthers.update({np.ndarray: (lambda x: x.nbytes)})


def bytewise_memoryview(x: Union[TYPE_INPUTDATA, TYPE_RETURNDATA]) -> memoryview:
    """Return a memoryview that is bytewise.

    Memory views have strides related to shape and known data types.
    This method removes that information and makes the view
    function bytewise.
    """
    view = memoryview(x)  # type: ignore
    return view.cast(format="B")
