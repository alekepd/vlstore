"""Simple binary object store based on SChunk."""
from typing import (
    Dict,
    List,
    Union,
    Optional,
    Iterable,
    Literal,
    overload,
    Final,
    Tuple,
    Any,
    TypeVar,
    Generic,
    ValuesView,
    KeysView,
    Iterator,
)
from math import ceil
from dataclasses import dataclass
from pathlib import Path
from os import PathLike
import blosc2  # type: ignore
from ._types import (
    TYPE_OUT,
    TYPE_RETURNDATA,
    TYPE_MINIMAL_RETURNDATA,
    TYPE_KEY,
    TYPE_INPUTDATA,
    T_OUT,
)
from .util import bytewise_memoryview

_TYPE_PATHCOMPAT = Union[str, PathLike]

DEFAULT_CHUNK_SIZE: Final = int(2**22)

_default_cparams = blosc2.cparams_dflts.copy()
_default_cparams["typesize"] = 1
_default_cparams["codec"] = blosc2.Codec.LZ4HC
# setting ["use_dict"] = 1 causes bugs.
_default_dparams = blosc2.dparams_dflts.copy()


def _create_default_schunk(
    *,
    filename: Optional[Path] = None,
    chunksize: int = DEFAULT_CHUNK_SIZE,
    cparams: Optional[Dict] = None,
    dparams: Optional[Dict] = None,
    contiguous: bool = True,
) -> blosc2.SChunk:
    args: Dict[str, Any] = {}
    args["chunksize"] = chunksize
    if cparams is None:
        cparams = _default_cparams
    args["cparams"] = cparams
    if dparams is None:
        dparams = _default_dparams
    args["dparams"] = dparams
    if filename is not None:
        args["urlpath"] = str(filename)
    args["contiguous"] = contiguous
    return blosc2.SChunk(**args)


@dataclass(frozen=True)
class Location:
    """Describes a contiguous portion in chunked storage."""

    start_block: int
    end_block: int
    block_size: int
    # this should be non-negative
    start_offset: int = 0
    # this should be non-positive
    end_offset: int = 0

    def __post_init__(self) -> None:
        """Validate set values."""
        if self.end_offset > 0 or (-self.end_offset >= self.block_size):
            raise ValueError(
                "end_offset must be non-positive and less in magnitude than block_size."
            )
        if self.start_offset < 0 or (self.end_offset >= self.block_size):
            raise ValueError(
                "end_offset must be non-negative and less than block_size."
            )

    @property
    def length(self) -> int:
        """Return the length of the selection."""
        return self.end - self.start

    @property
    def start(self) -> int:
        """Return the raw start location."""
        return self.block_size * self.start_block + self.start_offset

    @property
    def end(self) -> int:
        """Return the raw end location."""
        return self.block_size * self.end_block + self.end_offset

    @property
    def n_blocks(self) -> int:
        """Return the raw end location."""
        return self.end_block - self.start_block

    @classmethod
    def from_start_end(cls, start: int, end: int, block_size: int) -> "Location":
        """Create record from byte-based start and ends.

        Arguments:
        ---------
        start:
            Absolute (e.g., byte-wise) start location.
        end:
            Absolute (e.g., byte-wise) end location.
        block_size:
            Size of blocks. In above examples, in bytes.

        Results:
        -------
        Location instance.

        """
        raw_start_block, start_remainder = divmod(start, block_size)
        raw_end_block, end_remainder = divmod(end, block_size)
        return cls(
            start_block=raw_start_block,
            end_block=raw_end_block + ceil(end_remainder),
            block_size=block_size,
            start_offset=start_remainder,
            end_offset=block_size - end_remainder if end_remainder else 0,
        )


_T_KEY = TypeVar("_T_KEY", bound=TYPE_KEY)
_TUPLE_FORM = Tuple[_T_KEY, int, int]


class LocationIndex(Generic[_T_KEY]):
    """Key-store of Locations.

    A dictionary-like collection of Locations. Supports simple "serialization"
    methods into primitive types for storage.
    """

    def __init__(self, locs: Optional[Dict[_T_KEY, Location]] = None) -> None:
        """Initialize."""
        if locs is None:
            self.backing: Dict[_T_KEY, Location] = {}
        else:
            self.backing = locs

    def __setitem__(self, key: _T_KEY, value: Location) -> None:
        """Set item by key."""
        self.backing[key] = value

    def __getitem__(self, key: _T_KEY) -> Location:
        """Get item by key."""
        return self.backing[key]

    def __delitem__(self, key: _T_KEY) -> None:
        """Delete item by key."""
        del self.backing[key]

    def __contains__(self, key: _T_KEY) -> bool:
        """Check if key is present in backing."""
        return key in self.backing

    def __len__(self) -> int:
        """Return length of backing."""
        return len(self.backing)

    def __iter__(self) -> Iterator[_T_KEY]:
        """Iterate over keys in backing."""
        return iter(self.backing)

    def values(self) -> ValuesView:
        """Get values of underlying mapping."""
        return self.backing.values()

    def keys(self) -> KeysView:
        """Get keys of underlying mapping."""
        return self.backing.keys()

    def to_ordered_pairs(self) -> List[_TUPLE_FORM]:
        """Translate content into list of ordered pairs.

        First element of each tuple is the key, second is the absolute start of the
        entry, third is the absolute end of the entry.
        """
        return [(key, record.start, record.end) for key, record in self.backing.items()]

    @classmethod
    def from_ordered_pairs(
        cls, content: Iterable[_TUPLE_FORM], block_size: int
    ) -> "LocationIndex":
        """Create instance using list of tuples and known block size.

        First element of each tuple is the key, second is the absolute start of the
        entry, third is the absolute end of the entry.
        """
        formed = (
            (d[0], Location.from_start_end(start=d[1], end=d[2], block_size=block_size))
            for d in content
        )
        return cls(locs=dict(formed))


class SChunkStore:
    """Stores byte sequences using an underlying SChunk.

    As items are added, the SChunk is extended. Items are split and padded
    to stay compatible with the underlying chunk requirements, and stored
    values are encoded in the vlmeta for persistent self-describing SChunk-based
    storage.

    No methods are available to reclaim storage, but deleted data is set to zero
    which should be efficiently handled by underlying compression libraries.

    Attributes:
    ----------
    lookup:
        Dictionary mapping keys to Location instances; these instances determine
        where data has been stored.
    backing:
        SChunk instance that stores data in memory on on disk.

    """

    def __init__(
        self, location: Union[None, _TYPE_PATHCOMPAT, blosc2.SChunk] = None, **kwargs
    ) -> None:
        """Initialize BStore.

        Arguments:
        ---------
        location:
            If specified, the persistent location of the underlying schunk store.
            If it does not exist, it is created; if it already exists, it is opened
            read-only. If None, an in-memory store is used. If an Schunk instance,
            directly used as store. Note that in the last case limited validation
            on the instance is performed, and context manager behavior is limited.
        **kwargs:
            Passed to schunk creation if file/in-memory store is created. If opening
            and existing file, ignored.

        """
        self.lookup: LocationIndex[TYPE_KEY] = LocationIndex()
        # manage schunk store
        if location is None:
            # in-memory backing
            self.backing = _create_default_schunk(**kwargs)
        elif isinstance(location, blosc2.SChunk):
            self.backing = location
        else:
            # disk backing
            _p = Path(location).resolve()
            if _p.is_dir():
                raise ValueError(
                    "Please specify a (possible) file, not an existing directory."
                )
            elif _p.is_file():
                # need to pass more compression params?
                # keep this RO for now
                self.backing = blosc2.open(_p, mode="r")
            elif _p.parent.is_dir():
                self.backing = _create_default_schunk(filename=_p, **kwargs)
            else:
                raise ValueError("Could not use file location.")
        self.transfer_buffer = bytearray(self.chunksize)

    @property
    def chunksize(self) -> int:
        """Return chunksize of underlying Schunk instance."""
        return self.backing.chunksize

    @property
    def typesize(self) -> int:
        """Return typesize of underlying Schunk instance."""
        return self.backing.typesize

    @property
    def locations(self) -> List[Location]:
        """Return list of all used storage locations."""
        return list(self.lookup.values())

    def put(
        self,
        key: TYPE_KEY,
        value: TYPE_INPUTDATA,
        overwrite: bool = True,
        check: bool = True,
    ) -> None:
        """Put data in store.

        Arguments:
        ---------
        key:
            Value to use to retrieve data.
        value:
            Data to store.
        overwrite:
            If true, if there is already an entry for a the provided key, we disown
            the old entry and then store the new data. If false, we raise a ValueError.
        check:
            Checks to see if the allocated storage space storage matches the length of
            value.

        """
        # check to see if something already is present under this key
        already_exists = key in self.lookup
        if overwrite and already_exists:
            self.disown(key)
        elif (not overwrite) and already_exists:
            raise ValueError(f"Entry already exists for {key!r}.")

        # store value
        value_size = len(bytewise_memoryview(value))
        if value_size % self.backing.typesize != 0:
            raise ValueError("Storage object size not divisible by typesize.")
        prepped = self._prepare(value)
        remainder = value_size % self.chunksize
        fill_size = self.chunksize - remainder if remainder else 0
        location = Location(
            start_block=self.backing.nchunks,
            end_block=self.backing.nchunks + len(prepped),
            block_size=self.chunksize,
            end_offset=-fill_size,
        )
        if check and location.end - location.start != value_size:
            raise ValueError("Allocated slot is the wrong size for the chunk.")
        for c in prepped:
            self.backing.append_data(c)
        self.lookup[key] = location

    @overload
    def get(
        self,
        key: TYPE_KEY,
        *,
        out: Literal[None] = ...,
        return_location: Literal[False] = ...,
        method: Literal["slice", "chunk"] = ...,
    ) -> TYPE_MINIMAL_RETURNDATA:
        ...

    @overload
    def get(
        self,
        key: TYPE_KEY,
        *,
        out: T_OUT,
        return_location: Literal[False] = ...,
        method: Literal["slice", "chunk"] = ...,
    ) -> T_OUT:
        ...

    @overload
    def get(
        self,
        *,
        key: TYPE_KEY,
        out: Optional[TYPE_OUT] = ...,
        return_location: Literal[True],
        method: Literal["slice", "chunk"] = ...,
    ) -> Location:
        ...

    def get(
        self,
        key: TYPE_KEY,
        *,
        out: Optional[TYPE_OUT] = None,
        return_location: bool = False,
        method: Literal["slice", "chunk"] = "slice",
    ) -> Union[TYPE_RETURNDATA, Location, None]:
        """Get data from store.

        Arguments:
        ---------
        key:
            Key of content that will be retrieved.
        return_location:
            If true, instead of accessing data, we return the location
            of the data. This may later be transformed and passed to
            self.backing.get_slice in order to retrieve the data by another method.
            Else, we extract the data; see out for additional details.
        out:
            If return_location is False, we extract data from the store; out controls
            how this data is then returned. If out is not None, we deposit the data
            in the out object (See SChunk.get_slice for more information). If false,
            we return a bytes object with the data.
        method:
            Approach used to obtain data. "chunk" loads each chunk of data using
            the python interface, while "slice" uses a single call to the underlying
            blosc library. "slice" may be faster.

        Returns:
        -------
        Either a Tuple of 2 integers, a bytes object, or None. See return_location and
        out.

        """
        location = self.lookup[key]
        if return_location:
            return location
        if method == "slice":
            return self._slice_load(location=location, out=out)
        elif method == "chunk":
            return self._chunk_load(location=location, out=out)
        else:
            raise ValueError()

    @overload
    def _slice_load(
        self,
        location: Location,
        out: T_OUT,
    ) -> T_OUT:
        ...

    @overload
    def _slice_load(
        self,
        location: Location,
        out: Literal[None] = ...,
    ) -> bytes:
        ...

    def _slice_load(
        self, location: Location, out: Optional[TYPE_OUT] = None
    ) -> TYPE_RETURNDATA:
        item_start, start_remainder = divmod(location.start, self.typesize)
        item_end, end_remainder = divmod(location.end, self.typesize)
        if start_remainder > 0 or end_remainder > 0:
            raise ValueError("Entry is not typesize aligned.")
        if out is None:
            return self.backing.get_slice(item_start, item_end)
        else:
            # always return the data containing object
            self.backing.get_slice(item_start, item_end, out=out)
            return out

    @overload
    def _chunk_load(
        self,
        location: Location,
        out: T_OUT,
    ) -> T_OUT:
        ...

    @overload
    def _chunk_load(
        self,
        location: Location,
        out: Literal[None] = ...,
    ) -> bytearray:
        ...

    def _chunk_load(
        self, location: Location, out: Optional[TYPE_OUT] = None
    ) -> TYPE_RETURNDATA:
        if location.start_offset > 0:
            raise ValueError("non-zero start_offset not supported by chunked loading.")
        if out is None:
            out = bytearray(location.length)
        view = bytewise_memoryview(out)
        end = 0
        for write_index, chunk_index in enumerate(
            range(location.start_block, location.end_block - 1)
        ):
            start = write_index * location.block_size
            end = (write_index + 1) * location.block_size
            self.backing.decompress_chunk(chunk_index, dst=view[start:end])
        self.backing.decompress_chunk(location.end_block - 1, dst=self.transfer_buffer)
        if location.end_offset == 0:
            view[end:] = self.transfer_buffer[:]
        else:
            view[end:] = self.transfer_buffer[: location.end_offset]
        return out

    def fused_size(self, keys: Iterable[TYPE_KEY], *, presorted: bool = False) -> int:
        """Return size of buffer (in bytes) that a fused read would require."""
        if presorted:
            records = [self.lookup[x] for x in keys]
        else:
            # this could probably be made to have better scaling, we only need
            # to get the start and end
            records = sorted((self.lookup[x] for x in keys), key=lambda x: x.start)
        return records[-1].end - records[0].start

    @overload
    def fused_get(
        self,
        keys: Iterable[TYPE_KEY],
        *,
        presorted: bool = ...,
        out: T_OUT,
    ) -> Tuple[T_OUT, List[slice]]:
        ...

    @overload
    def fused_get(
        self,
        keys: Iterable[TYPE_KEY],
        *,
        presorted: bool = ...,
        out: Literal[None] = ...,
    ) -> Tuple[TYPE_MINIMAL_RETURNDATA, List[slice]]:
        ...

    def fused_get(
        self,
        keys: Iterable[TYPE_KEY],
        *,
        presorted: bool = False,
        out: Optional[TYPE_OUT] = None,
    ) -> Tuple[TYPE_RETURNDATA, List[slice]]:
        """Read multiple entries using one call.

        Note that this operation returns views on a collective buffer. This collective
        buffer may be larger than all of the items request due to padding or requesting
        a non-contiguous set of records.

        Arguments:
        ---------
        keys:
            Keys to look up values for.
        presorted:
            Whether the given keys corresponds to entries which are stored at increasing
            offsets. Do not set to true if you do know that this means.
        out:
            Buffer to read into; a single buffer is used for all items.  The size of
            this buffer can be obtained using the fused_size method.

        Returns:
        -------
        2-tuple: First element is a bytes object con

        """
        keys = list(keys)
        if presorted:
            records = [self.lookup[x] for x in keys]
        else:
            # this could probably be made to have better scaling, we only need
            # to get the start and end
            records = sorted((self.lookup[x] for x in keys), key=lambda x: x.start)
        first = records[0]
        last = records[-1]
        proxy_location = Location(
            start_block=first.start_block,
            end_block=last.end_block,
            block_size=first.block_size,
            start_offset=first.start_offset,
            end_offset=last.end_offset,
        )
        collective_data = self._slice_load(location=proxy_location, out=out)
        slices: List[slice] = []
        offset = first.start
        for entry in (self.lookup[k] for k in keys):
            slices.append(slice(entry.start - offset, entry.end - offset, None))

        return (collective_data, slices)

    def disown(self, key: TYPE_KEY, zero: bool = False) -> None:
        """Remove item from internal records and optionally wipe storage.

        Arguments:
        ---------
        key:
            Descriptor of item to disown.
        zero:
            If true, we write over the disowned area with zeros. This may
            help memory usage if compression is on, but incurs a disk write.

        """
        location = self.get(key=key, return_location=True)
        item_start = location.start * self.typesize
        item_end = location.start * self.typesize
        if zero:
            # bracket indexing on schunks uses counts based on typesize.
            # The syntax however allows a single i/o request to be made to
            # the SChunk library, which may be faster.

            # for a large object, this might be a costly temporary memory allocation.
            self.backing[item_start, item_end] = bytes(
                (item_end - item_start) * self.typesize
            )
        del self.lookup[key]

    def _init_buffer(self) -> None:
        """Initialize buffer for transfer.

        When objects are written to the store, they may not fill up the chunksize
        of the store. In this case, we create a chunk of correct size and fill it up
        partially. This method is called right before we put out data in said chunk.

        Currently, it just zeros the buffer, which should make compression perform
        better.

        """
        # zero out transfer buffer, this seems to be the most direct way
        # to do so to a bytearray.
        bytearray.__init__(self.transfer_buffer, len(self.transfer_buffer))

    def _prepare(
        self,
        source: TYPE_INPUTDATA,
    ) -> List[memoryview]:
        """Transform a buffer to a series of buffers of a given size.

        Note that the final buffer may be padded with zeros, and that returned
        buffers may refer to the memory underlying the original object.

        Arguments:
        ---------
        source:
            Buffer to break into parts.

        Returns:
        -------
        List of memoryview instances. All but the final instance will be a view of the
        original objects memory.

        """
        source_view = bytewise_memoryview(source)
        source_size = len(source_view)
        n_full_buffers, remainder = divmod(source_size, self.chunksize)
        prepped: List[memoryview] = []
        transfer_index = -1
        for transfer_index in range(n_full_buffers):
            selection = slice(
                transfer_index * self.chunksize, (transfer_index + 1) * self.chunksize
            )
            prepped.append(source_view[selection])
        # if we need to make a new padded chunk for the last part of the data.
        if remainder > 0:
            # zero out our existing buffer
            self._init_buffer()
            subsize = source_size - (transfer_index + 1) * self.chunksize
            self.transfer_buffer[0:subsize] = source_view[
                (transfer_index + 1) * self.chunksize : source_size
            ]
            prepped.append(memoryview(self.transfer_buffer))
        return prepped

    def __delitem__(self, key: TYPE_KEY) -> None:
        """Remove item from record.

        Note that this does not reclaim any memory directly, it just removes the
        record of the data and zeroes out the region. Compression may make this action
        take up less physical memory or disk space.
        """
        self.disown(key, zero=True)

    def __getitem__(self, key: TYPE_KEY) -> TYPE_MINIMAL_RETURNDATA:
        """Retrieve item from store.

        Similar to get, but with fewer options.
        """
        return self.get(key)

    def __setitem__(self, key: TYPE_KEY, newvalue: TYPE_INPUTDATA) -> None:
        """Deposit bytes instance into the store."""
        self.put(key=key, value=bytewise_memoryview(newvalue), overwrite=True)

    def __iter__(self) -> Iterable[TYPE_KEY]:
        """Iterate over the keys of stored items."""
        return iter(self.lookup)

    def __len__(self) -> int:
        """Return number of stored items.

        Note that this is not related to the size of stored items.
        """
        return len(self.lookup)
