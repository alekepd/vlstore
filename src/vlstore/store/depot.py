"""Storage for serializable objects."""

from typing import (
    TypeVar,
    Generic,
    Optional,
    Iterable,
    List,
)
from .chunkstore import SChunkStore
from ..serialize import Codec
from .util import bytewise_memoryview
from ._types import (
    TYPE_KEY,
    TYPE_OUT,
)

_T = TypeVar("_T")


class Depot(Generic[_T]):
    """Storage of serializable objects.

    Thin wrapper over underlying store methods. Once we have flatpack support
    and can optimize we can reconsider its role. It is cleaner to leave serialization
    outside of byte methods, and this method may be useful for storing metadata outside
    the keystore.
    """

    def __init__(
        self,
        codec: Codec[_T],
        backing: Optional[SChunkStore] = None,
        initial_buffer_size: int = 0,
        recycle_buffer: bool = False,
    ) -> None:
        """Initialize."""
        if backing is None:
            self.backing = SChunkStore()
        else:
            self.backing = backing
        self.codec = codec
        self.recycle_buffer = recycle_buffer
        if recycle_buffer:
            self._buffer: Optional[memoryview] = memoryview(
                bytearray(initial_buffer_size)
            )
        else:
            self._buffer = None

    def put(self, key: TYPE_KEY, value: _T) -> None:
        """Serialize and store object."""
        data = self.codec.pack(value)
        self.backing.put(key, data)

    def get(self, key: TYPE_KEY, **kwargs) -> _T:
        """Retrieve and deserialize object."""
        # should implement buffer to read into.
        data = self.backing.get(key, **kwargs)
        return self.codec.unpack(data)

    def fused_get(
        self,
        keys: Iterable[TYPE_KEY],
        *,
        presorted: bool = False,
        buffer: Optional[TYPE_OUT] = None,
    ) -> List[_T]:
        """Retrieve and deserialize multiple objects."""
        if not presorted:
            keys = sorted(keys)
        if buffer is None:
            size = self.backing.fused_size(keys, presorted=True)
            view = self._get_buffer(size)
        else:
            view = bytewise_memoryview(buffer)
        _, slices = self.backing.fused_get(keys, presorted=True, out=view)
        to_return = []
        for x in slices:
            to_return.append(self.codec.unpack(view[x]))
        return to_return

    def _get_buffer(self, size: int) -> memoryview:
        # we can incorporate some memory guards into this,
        # maybe based on gc reference counts
        if not self.recycle_buffer:
            return memoryview(bytearray(size))
        if len(self._buffer) < size:  # type: ignore
            self._buffer = memoryview(bytearray(size))
        return self._buffer[0:size]  # type: ignore
