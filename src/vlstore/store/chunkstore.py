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
    Callable,
)
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
_default_cparams["codec"] = blosc2.Codec.ZSTD
_default_cparams["clevel"] = 1

# setting ["use_dict"] = 1 causes bugs.
_default_dparams = blosc2.dparams_dflts.copy()


def _create_default_schunk(
    *,
    filename: Optional[Path] = None,
    chunksize: int = DEFAULT_CHUNK_SIZE,
    cparams: Optional[Dict] = None,
    dparams: Optional[Dict] = None,
    contiguous: bool = True,
    meta: Optional[Dict[Union[bytes, str], Any]] = None,
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
    if meta is not None:
        args["meta"] = meta
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

    def block_split(self) -> List["Location"]:
        """Split location according to block boundaries.

        If a given location spans multiple blocks, this function breaks that location
        into parts, where each part only lives on a single block. Note that end_block
        of the derived single-block locations will not be the same as start_block, as
        the end index is exclusive.

        """
        to_return: List[Location] = []
        # if already a single block location, return list of self.
        if self.start_block == (self.end_block - 1):
            return [self]
        else:
            # we are on more than one block

            # First block defined to be start to first block mark
            # Note that this _might_ be partial.
            partial_first_chunk = self.from_start_end(
                self.start,
                (self.start_block + 1) * self.block_size,
                block_size=self.block_size,
            )
            to_return.append(partial_first_chunk)
        # middle blocks are full by definition, so we can quickly define them
        # note that this loop may be empty, that is fine.
        for chunk in range(self.start_block + 1, self.end_block - 1):
            full_chunk = Location(
                start_block=chunk, end_block=chunk + 1, block_size=self.block_size
            )
            to_return.append(full_chunk)
        # last block is defined to be the penultimate block to parent end point
        # Note that this _might_ be partial.
        partial_last_chunk = Location.from_start_end(
            start=(self.end_block - 1) * self.block_size,
            end=self.end,
            block_size=self.block_size,
        )
        to_return.append(partial_last_chunk)
        return to_return

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

    def __len__(self) -> int:
        """Return raw (not block-based) length."""
        return self.end - self.start

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
            end_block=raw_end_block + 1 if end_remainder else raw_end_block,
            block_size=block_size,
            start_offset=start_remainder,
            end_offset=end_remainder - block_size if end_remainder else 0,
        )


_T = TypeVar("_T")


def _iter_max(content: Iterable[_T], call: Callable[[_T], Any]) -> _T:
    """Get maximum of iterable.

    Arguments:
    ---------
    content:
        Iterable to go through. If zero length, exception is raised.
    call:
        Applied to each item before comparison is done.

    Returns:
    -------
    Biggest item (_not_ the output of call on that object).

    """
    _it = iter(content)
    try:
        best = next(_it)
    except StopIteration as e:
        raise ValueError("Empty sequence") from e
    best_val = call(best)
    for new_val, new in ((call(x), x) for x in _it):
        if new_val > best_val:
            best = new
            best_val = new_val
    return best


def _iter_min(content: Iterable[_T], call: Callable[[_T], Any]) -> _T:
    """Get minimum of iterable.

    Arguments:
    ---------
    content:
        Iterable to go through. If zero length, exception is raised.
    call:
        Applied to each item before comparison is done.

    Returns:
    -------
    Smallest item (_not_ the output of call on that object).

    """
    _it = iter(content)
    try:
        best = next(_it)
    except StopIteration as e:
        raise ValueError("Empty sequence") from e
    best_val = call(best)
    for new_val, new in ((call(x), x) for x in _it):
        if new_val < best_val:
            best = new
            best_val = new_val
    return best


def _chunk_memoryview(sizes: List[int], target: memoryview) -> List[memoryview]:
    """Split memoryview into partial memory views of given sizes."""
    # assume length of target is not 0
    if sum(sizes) != len(target):
        raise ValueError("Chunk sizes do not correspond to target length.")
    to_return: List[memoryview] = []
    start = 0
    for length in sizes:
        end = start + length
        to_return.append(target[start:end])
        start = end
    return to_return


_T_KEY = TypeVar("_T_KEY", bound=TYPE_KEY)
_TUPLE_FORM = Tuple[_T_KEY, int, int]


_TYPE_NO_CROSS_ALIGNMENT = Literal["no_cross"]
_TYPE_CONTIGUOUS_ALIGNMENT = Literal["contiguous"]
_TYPE_START_ALIGNMENT = Literal["start"]
TYPE_ALIGNMENT = Union[
    _TYPE_NO_CROSS_ALIGNMENT, _TYPE_CONTIGUOUS_ALIGNMENT, _TYPE_START_ALIGNMENT
]
NOCROSS_ALIGNMENT: Final = "no_cross"
CONTIGUOUS_ALIGNMENT: Final = "contiguous"
START_ALIGNMENT: Final = "start"


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
        self._last: Optional[Location] = None
        self._first: Optional[Location] = None

    def _reset_firstlast(self) -> None:
        self._last = None
        self._first = None

    @property
    def last(self) -> Location:
        """Return Location with highest end value."""
        if self._last is None:
            self._last = _iter_max(self.backing.values(), call=lambda x: x.end)
        return self._last

    @property
    def first(self) -> Location:
        """Return Location with lowest start value."""
        if self._first is None:
            self._first = _iter_min(self.backing.values(), call=lambda x: x.start)
        return self._first

    def __setitem__(self, key: _T_KEY, value: Location) -> None:
        """Set item by key."""
        self.backing[key] = value
        if self._first is None or value.start < self._first.start:
            self._first = value
        if self._last is None or value.end > self._last.end:
            self._last = value

    def __getitem__(self, key: _T_KEY) -> Location:
        """Get item by key."""
        return self.backing[key]

    def __delitem__(self, key: _T_KEY) -> None:
        """Delete item by key."""
        del self.backing[key]
        self._reset_firstlast()

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

    def plan(
        self,
        size: int,
        alignment: TYPE_ALIGNMENT = NOCROSS_ALIGNMENT,
        block_size: Optional[int] = None,
    ) -> Location:
        """Return location corresponding to free space.

        Arguments:
        ---------
        size:
            Size of requested space.
        alignment:
            Strategy used to allocate given space. Possible options are given in
            module variables NOCROSS_ALIGNMENT, CONTIGUOUS_ALIGNMENT, and
            START_ALIGNMENT. The first allocates space for new entries in the most
            compact way such that no record crosses a chunk (not block) boundary.
            The second aligns records contiguously, and the third only places records
            at the start of each chunk.
        block_size:
            The size of block to use in the allocation. If Locations are already
            present (i.e., length > 0), setting None will use the block size
            in the last value. If empty, this must be specified.

        Results:
        -------
        Location. Note that this location is not recorded in the instance.

        """
        if block_size is None:
            try:
                block_size = self.last.block_size
            except ValueError as e:
                raise ValueError("When empty, block_size must be specified.") from e

        if not self.backing:
            # we have no records
            start = 0
        elif alignment == NOCROSS_ALIGNMENT:
            # if there is enough space in the current block for the record, put it
            # there; otherwise skip forward to the next block.
            if abs(self.last.end_offset) > size:
                start = self.last.end
            else:
                start = self.last.end_block * self.last.block_size
        elif alignment == CONTIGUOUS_ALIGNMENT:
            start = self.last.end
        elif alignment == START_ALIGNMENT:
            start = self.last.end_block * self.last.block_size
        else:
            raise ValueError(f"Invalid allocation alignment strategy: {alignment}.")

        return Location.from_start_end(start, start + size, block_size=block_size)

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


class _BoundChunkWriter:
    """Cached chunk writer for SChunk objects.

    This object is persistently associated with a selected chunk in an SChunk index.
    Calling flush explicitly, or resassociating it, causes the stored data to be flushed
    to the SChunk. Internal buffers remain allocated during reassociation, but are
    wiped.

    Current, it will only represent a chunk that is already allocated or immediately
    after what is allocated.

    Ideally, would support memoryview, but this is not simple before py 3.12. Instead,
    use memoryview(ob.buffer) or access the buffer attribute directly.

    Important attributes:
    --------------------
    backing:
        SChunk instance that is modified.
    buffer:

    index:
        Integer specifying which chunk we are associated with or None; None means
        we are unassociated.

    """

    def __init__(
        self, backing: blosc2.SChunk, index: Optional[int] = None, copy: bool = False
    ) -> None:
        """Initialize.

        Arguments:
        ---------
        backing:
            Schunk instance to write to.
        index:
            Either an integer specifying a chunk or None. If none, the instance is
            considered unbound, and calling flush will not write anything.
        copy:
            Passed to schunk.update_data if modifying an existing chunk.

        """
        self.index: Optional[int] = index
        self.backing = backing
        self.buffer = memoryview(bytearray(self.backing.chunksize))
        self.copy = copy
        self.index = None
        self._zeroes = memoryview(bytes(self.backing.chunksize))
        if index is not None:
            self.associate(index)

    def flush(self) -> None:
        """Flush cached data to disk.

        Sets self.index to None, making the object no longer associated with
        any chunk. If non-associated, this call does nothing.
        """
        if self.index is None:
            return
        # Check to see if we should update or append the data.
        if self.index < self.backing.nchunks:
            self.backing.update_data(self.index, self.buffer, copy=self.copy)
        elif self.index == self.backing.nchunks:
            self.backing.append_data(self.buffer)
        else:
            # this should not happen because index is validated at construction.
            raise ValueError(
                "Attempting to write block that is not a previous or next block."
            )
        self.index = None

    def associate(self, index: int) -> None:
        """Associate with a new chunk.

        If the new index is different than the internal index, the current buffer
        is flushed to the underlying SChunk before reassociating. If the index
        is the same as the internal index, no flushing is done.

        If the new chunk already exists in the SCHunk, its content is retrieved
        and placed in the buffer. If the new chunk does not exist, but is only
        one above the existing chunk range, the array is initialized with zeroes.
        If larger, a ValueError is raised.

        Arguments:
        ---------
        index:
            New chunk index to associate with.

        Returns:
        -------
        None

        """
        if self.index == index:
            return

        if self.index is not None:
            self.flush()
            # flush must happen before changing index

        self.index = index

        if self.index < self.backing.nchunks:
            # we are using a chunk in the backing, get its contents
            self.backing.decompress_chunk(self.index, dst=self.buffer)
        elif self.index == self.backing.nchunks:
            # we are using a chunk not in the backing, init with zeroes
            self.buffer[:] = self._zeroes
        else:
            raise ValueError(
                "Cannot be associated with a block that is not a previous or "
                "next block."
            )

    def write_through(self, index: int, content: memoryview) -> None:
        if index == self.index:
            # this operation will invalidate whatever content was present
            self.index = None
            self.buffer[:] = self._zeroes
        _write_block_to_schunk(
            index=index, content=content, schunk=self.backing, copy=self.copy
        )


def _write_block_to_schunk(
    index: Optional[int],
    content: memoryview,
    schunk: blosc2.SChunk,
    copy: bool = False,
) -> None:
    """Write a full block/chunk to SChunk file.

    If the specified chunk location exists, the its content is overwritten. If
    the specified index is one more than what is present in the SChunk, the new
    data is appended. If the specified index is larger, a ValueError is raised.

    Arguments:
    ---------
    index:
        Block index to be (over)written. See function description.
    content:
        memoryview of content to place in block location. Must be the correct size
        to exactly fill the entire block.
    schunk:
        SChunk instance to write to.
    copy:
        Passed to update_data if an update operation is internally selected.

    Returns:
    -------
    None

    """
    # e.g. if nchunks if 4
    # we have chunks 0 1 2 3
    if index is None:
        return
    if index < schunk.nchunks:
        schunk.update_data(index, content, copy=copy)
    elif index == schunk.nchunks:
        schunk.append_data(content)
    else:
        raise ValueError(
            "Attempting to write block that is not a previous or next block."
        )


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

    VLMETA_SAVEDLOOKUP: Final = "saved_lookup"
    # generated via uuid4
    META_MAGIC: Final = "magic"
    # identifies schunk file as conforming to format
    MAGIC: Final = b"\xa4\x9a3g\xcf}D\xd1\xb0\x97\xe0Q\x05\x836\xeb"

    def __init__(
        self,
        location: Union[None, _TYPE_PATHCOMPAT, blosc2.SChunk] = None,
        alignment: TYPE_ALIGNMENT = NOCROSS_ALIGNMENT,
        **kwargs,
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
        alignment:
            Strategy used to allocate given space. Possible options are given in
            module variables NOCROSS_ALIGNMENT, CONTIGUOUS_ALIGNMENT, and
            START_ALIGNMENT. The first allocates space for new entries in the most
            compact way such that no record crosses a chunk (not block) boundary.
            The second aligns records contiguously, and the third only places records
            at the start of each chunk.
        **kwargs:
            Passed to schunk creation if file/in-memory store is created. If opening
            and existing file, ignored.

        """
        self.lookup: LocationIndex[TYPE_KEY] = LocationIndex()
        self.alignment = alignment
        options = kwargs.copy()
        if "meta" in options:
            options["meta"].update({self.META_MAGIC: self.MAGIC})
        else:
            options["meta"] = {self.META_MAGIC: self.MAGIC}

        # manage schunk store
        self.read_only = False
        if location is None:
            # in-memory backing
            self.backing = _create_default_schunk(**options)
        elif isinstance(location, blosc2.SChunk):
            if len(kwargs) > 0:
                raise ValueError(
                    "Schunk directly passed but construction options are also present."
                )
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
                self.read_only = True
                self.backing = blosc2.open(_p, mode="r")
                assert self.backing.meta[self.META_MAGIC] == self.MAGIC
                self.lookup = LocationIndex.from_ordered_pairs(
                    self.backing.vlmeta[self.VLMETA_SAVEDLOOKUP],
                    block_size=self.chunksize,
                )
            elif _p.parent.is_dir():
                self.backing = _create_default_schunk(filename=_p, **options)
            else:
                raise ValueError("Could not use file location.")

        # must happen after backing is initialized

        # buffer for chunk-wise reading
        self._transfer_buffer = memoryview(bytearray(self.chunksize))

        # buffered object for writing
        self._in_progress_block = _BoundChunkWriter(backing=self.backing)

        # used to quickly zero out other buffers
        self._zero_buffer = memoryview(bytes(self.chunksize))

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
        if self.read_only:
            raise ValueError("Backing is read-only.")
        # check to see if something already is present under this key
        already_exists = key in self.lookup
        if overwrite and already_exists:
            self.disown(key)
        elif (not overwrite) and already_exists:
            raise ValueError(f"Entry already exists for {key!r}.")
        # check value length
        value_view = bytewise_memoryview(value)
        value_size = len(value_view)
        if value_size == 0:
            raise ValueError("Cannot store length-0 bytes.")
        # get required locations we will write to
        location = self.lookup.plan(
            value_size, alignment=self.alignment, block_size=self.chunksize
        )
        location_chunks = location.block_split()
        # break data into those chunk sizes
        chunk_views = _chunk_memoryview(
            sizes=[len(x) for x in location_chunks], target=value_view
        )
        # write the data
        self._write_views(views=chunk_views, locations=location_chunks)

        if check and location.end - location.start != value_size:
            raise ValueError("Allocated slot is the wrong size for the chunk.")

        # record the location
        self.lookup[key] = location

    def _write_views(
        self, views: Iterable[memoryview], locations: Iterable[Location]
    ) -> None:
        """Write views to disk at given locations.

        Arguments:
        ---------
        views:
            memoryview instances that will be written. They must match in size
            to the entries in locations.
        locations:
            Location instances giving where writes will happen. All entries should
            occupy 1 chunk, sorted from low to high start points, and must be
            contiguous: there must be no gap between the end of one instance and
            the start of the next.

        Returns:
        -------
        None

        Notes:
        -----
        Writing is the most complex operation in this class. Data must be broken
        up and written obeying chunk sizes; furthermore, to avoid fragmentation,
        we must cache non-full chunks _between_ writes. Chunking is performed before
        this function, and this function takes care of writing/caching. This function
        assumes that we are only ever adding blocks to the end of the underlying schunk.
        Due to caching, after this function exist data may not yet be written to the
        underlying schunk; to be sure, call self._flush_block_cache().

        """
        # First and last views need special treatment. We assume there is always at
        # least one view.
        working_locations = list(locations)
        working_views = list(views)

        # Check is the first chunk is a partial chunk, which needs special treatment
        if len(working_views[0]) != self.chunksize:
            # remove said entry so we don't double process
            chunk = working_views.pop(0)
            chunk_location = working_locations.pop(0)

            # associate writer with chunk
            # this may carry over older data that is currently cached
            self._in_progress_block.associate(chunk_location.start_block)

            buffer_end_mark = (
                chunk_location.end_offset if chunk_location.end_offset else None
            )
            # store content in buffered writer
            self._in_progress_block.buffer[
                chunk_location.start_offset : buffer_end_mark
            ] = chunk
            # flush writer if we are at the end of the block
            if buffer_end_mark is None:
                self._in_progress_block.flush()

        # Continue if there are further chunks to write.
        if working_views:
            # if we are writing more blocks, we its easiest to flush
            # the cache first. This should happen automatically as well.
            self._in_progress_block.flush()

            # Deal with full-size middle chunk
            # if we have no middle chunks, this loop will do nothing
            for chunk, chunk_location in zip(
                working_views[:-1], working_locations[:-1]
            ):
                self._in_progress_block.write_through(
                    index=chunk_location.start_block, content=chunk
                )

            # deal with final chunk, need to consider if it needs to be padded.
            chunk = working_views[-1]
            chunk_location = working_locations[-1]
            if len(chunk) == self.chunksize:
                self._in_progress_block.write_through(
                    index=chunk_location.start_block, content=chunk
                )
            else:
                self._in_progress_block.associate(index=chunk_location.start_block)
                self._in_progress_block.buffer[: len(chunk)] = chunk

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
        # if we are looking up content that is currently in the cached writer,
        # flush the cache
        if (location.end_block - 1) == self._in_progress_block.index:
            self._in_progress_block.flush()
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
    ) -> Union[T_OUT, memoryview]:
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
        _ts = self.typesize
        start_quotient, start_remainder = divmod(location.start, _ts)
        end_quotient, end_remainder = divmod(location.end, _ts)
        corrected_end = end_quotient + 1 if end_remainder else end_quotient
        if out is None:
            out = self.backing.get_slice(start_quotient, corrected_end)
        else:
            # always return the data containing object
            self.backing.get_slice(start_quotient, corrected_end, out=out)
        if start_remainder == 0 and end_remainder == 0:
            return out
        else:
            end_offset = end_remainder - _ts if end_remainder else None
            buffer = bytewise_memoryview(out)[start_remainder:end_offset]
            assert len(buffer) == len(location)
            return buffer

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
        self.backing.decompress_chunk(location.end_block - 1, dst=self._transfer_buffer)
        if location.end_offset == 0:
            view[end:] = self._transfer_buffer[:]
        else:
            view[end:] = self._transfer_buffer[: location.end_offset]
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
        # if we are looking up content that is currently in the cached writer,
        # flush the cache
        if (proxy_location.end_block - 1) == self._in_progress_block.index:
            self._in_progress_block.flush()
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

    def _write_lookup(self) -> None:
        self.backing.vlmeta[self.VLMETA_SAVEDLOOKUP] = self.lookup.to_ordered_pairs()

    def close(self) -> None:
        """Write metadata to file.

        Other writes _should_ already be synchronous.
        """
        # flush any chunk we are in the middle of creating
        self._in_progress_block.flush()

        # write metadata
        self._write_lookup()

        # should some file be closed? blosc2 documents seem to
        # say no. Perhaps all writes are synchronous.

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
        self._transfer_buffer[:] = self._zero_buffer

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

    def __enter__(self) -> "SChunkStore":
        """Return self as open has already occurred."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> Literal[False]:
        """Close file and ignore exceptions."""
        if not self.read_only:
            self.close()
        return False
