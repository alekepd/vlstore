"""Implements a block-shuffled batch loader for Depot instances.

Functionality is exposed via BlockShuffleBatch.
"""
from typing import (
    Dict,
    TypeVar,
    Generic,
    List,
    Iterator,
    Optional,
    Generator,
    Iterable,
)
from random import shuffle as rshuffle
import warnings
from itertools import chain, cycle, islice
from ..store import Depot, LocationIndex, TYPE_KEY

_T_KEY = TypeVar("_T_KEY", bound=TYPE_KEY)
_T = TypeVar("_T")
_T_0 = TypeVar("_T_0")


def block_ownership(
    index: LocationIndex[_T_KEY],
    no_sharing: bool = False,
    sorted: bool = False,
) -> Dict[int, List[_T_KEY]]:
    """Determine which blocks in a LocationIndex contain which items.

    Note that an entry may be present in multiple block entries if no_sharing
    is not True.

    Arguments:
    ---------
    index:
        LocationIndex to process
    no_sharing:
        If true, if any item exists on more than one block,
        a ValueError is raised.
    sorted:
        Whether to sort the individual lists in the return value by start attribute.

    Returns:
    -------
    Dictionary mapping block index to lists of keys.

    """
    block_mapping: Dict[int, List[_T_KEY]] = {}
    for k in index.keys():
        loc = index[k]
        blocks = range(loc.start_block, loc.end_block)
        if len(blocks) > 1 and no_sharing:
            raise ValueError(
                "Found record spanning multiple blocks, but no_sharing "
                "is set to True."
            )
        for b in blocks:
            storage: List[_T_KEY] = block_mapping.get(b, [])
            storage.append(k)
            block_mapping[b] = storage
    if sorted:
        for v in block_mapping.values():
            v.sort(key=lambda x: index[x].start)
    return block_mapping


def batched(source: Iterable[_T], size: int) -> Iterator[List[_T]]:
    """Break iterable into chunks of a given size.

    The last chunk may be smaller.

    Arguments:
    ---------
    source:
        Source iterable to draw values from.
    size:
        Size of chunks served. Last chunk may be smaller.

    Returns:
    -------
    Generator that serves chunks of data.

    Notes:
    -----
    This routine will likely break in-out connections in chains of generators
    as it does not use yield from, but it is unclear how to make it do so.

    """
    if size < 1:
        raise ValueError("size must be larger than 0")
    it = iter(source)
    while batch := list(islice(it, size)):
        try:
            yield batch
        except StopIteration:
            pass


class BlockShuffleBatch(Generic[_T]):
    """Shuffles and serves data from a Depot.

    The Depot instance is partially read, shuffled, batched, and served;  this
    process is repeated to serve all data present. The data is not reordered
    with a uniform probability because of the partial reads, but may be a good
    approximation.

    For example, if read_size is 3, 3 blocks are first read from the Depot.
    The data from these 3 blocks is shuffled, batched, and served via
    iteration.  Once depleted, the next 3 blocks are read. Care is taken such
    that the batch size does not need to divide the read_size multiplied by the
    block size, and only the final batch may be smaller than the requested size.

    Important:
    ---------
    Persistent memory buffers used to read data from the Depot. If the Depot
    deserialization creates views of the supplied buffer (e.g., when using
    FlatBuffers instances), then data should be consumed as it is produced during
    iteration before obtaining the next batch.

    If more batches are needed at once, then the n_buffers init parameter can
    be increased, but the exact lifetime can be hard to predict. If
    deserialization does not create a view, then n_buffers can be set to 1 and
    served instances will remain valid.

    Attributes:
    ----------
    read_size:
        The number of blocks to load at once. Larger values require more memory,
        but more faithfully shuffle the data.
    batch_size:
        Size of batches served. Must be less than the smallest abstracted block size,
        see end_block_merge.
    backing:
        Depot instance to draw data from.
    shuffle:
        Whether to shuffle data during iteration. This is usually only false for testing
        purposes.
    end_block_merge:
        Often the final block in a Depot is only partially occupied, which can create
        aggressive limits on batch_size. When this option is true, the final block
        data is merged with the penultimate block to avoid this situation. This may
        increase memory usage by increasing the size of read buffers.
    entered:
        This object should not be iterated over when it is in the middle of an
        iteration. This behavior is controlled by the entered flag. If an iteration
        failure occurs, entered may be incorrectly set to True, stopping usage.
        It may manually then be set to False.

    Important methods:
    -----------------
    stream:
        Provides individual entries during iteration. Entries may become invalid
        after n_buffers*read_size iterations.
    __iter__:
        Provides lists of entries of size batch_size during iteration. Batches
        may become invalid at the same rate as entries found via stream; it is
        recommended that each batch is used before the next batch is requested.

    """

    def __init__(
        self,
        read_size: int,
        batch_size: int,
        backing: Depot[_T],
        buffer_size: Optional[int] = None,
        shuffle: bool = True,
        end_block_merge: bool = True,
        n_buffers: int = 2,
    ) -> None:
        """Initialize.

        See class description for more information.

        Arguments:
        ---------
        read_size:
            The number of blocks read from storage for each write session.
        batch_size:
            The number of items served during iteration.
        backing:
            Depot providing data to load.
        buffer_size:
            Size of buffers to create during loading. If None, the maximum possible read
            size is used; this is only set to values other than None during debugging.
        shuffle:
            Whether to shuffle data when serving; this is usually only False when
            testing. Shuffle changes the order of blocks read and the order of frames
            in between reads.
        end_block_merge:
            Whether to merge the final block in to penultimate block for blocked
            reading; increases acceptable batch_size but increases memory usage.
        n_buffers:
            Number of buffers to cycle through when reading. Larger numbers increase
            memory usage, but allow served items to stay alive longer. It is simplest
            to set this to 2 and then consume each batch as it is served.

        """
        self.read_size = read_size
        self.backing = backing
        self.shuffle = shuffle
        self.end_block_merge = end_block_merge
        self.ownership = block_ownership(backing.lookup, no_sharing=True, sorted=True)
        self.entered = False
        sizes = [
            self.backing.fused_buffer_size(kg, presorted=True)
            for kg in self.key_groups()
        ]
        if buffer_size is None:
            self.buffer_size = max(sizes)
        else:
            self.buffer_size = buffer_size
        smallest_group = min(len(x) for x in self.key_groups())
        if batch_size > smallest_group:
            raise ValueError(
                f"batch_size ({batch_size}) cannot be larger than"
                f" the smallest loading group: {smallest_group}"
            )
        self.batch_size = batch_size
        if n_buffers < 1:
            raise ValueError("n_buffers must be at least 1.")
        elif n_buffers < 2:
            warnings.warn(
                "Only using one buffer; if served objects are views, this may result"
                "in corruption. Try setting n_buffers to 2 or larger.",
                stacklevel=1,
            )
        # create n_buffers lists of memory views. Each can hold a full read iteration
        # on n_blocks.
        self._buffers: List[List[memoryview]] = [
            [memoryview(bytearray(self.buffer_size)) for _ in range(read_size)]
            for _ in range(n_buffers)
        ]

    def key_groups(
        self,
        shuffle: bool = False,
    ) -> List[List[TYPE_KEY]]:
        """Return groups of keys.

        This function defines the blocks of records that form the atomic read operations
        used in other routines. Currently, these blocks are primarily just the blocks
        in underlying blocked storage; the exception is the final block, which may be
        merged with the penultimate block. This is done to address the case where the
        final block is very small.

        Arguments:
        ---------
        shuffle:
            Whether to shuffle the ordering of the groups. The items inside each group
            are not shuffled.

        Returns:
        -------
        List of Lists of keys.

        """
        if self.end_block_merge:
            occupied_blocks = list(self.ownership.keys())
            # highest block index
            first = max(occupied_blocks)
            occupied_blocks.pop(first)
            # second highest block index
            second = max(occupied_blocks)
            kgroups = [v for k, v in self.ownership.items() if k not in [first, second]]
            # This relies on the original values being sorted, and second coming earlier
            # than first.
            kgroups.append(self.ownership[second] + self.ownership[first])
        else:
            kgroups = list(self.ownership.values())
        if shuffle:
            rshuffle(kgroups)
        return kgroups

    def _server(self, shuffle: bool) -> Generator[List[_T], List[memoryview], None]:
        """Transform lists of memoryviews into lists of data entries.

        This is a generator that both accepts and yields values. The first send value
        is None; after that, lists of memoryviews are accepted. These memoryviews
        are used as read buffers in fused_get operations; the output is presented via
        yield.

        Arguments:
        ---------
        shuffle:
            Whether to shuffle items. Shuffling is performed first in the ordering of
            blocks read, and then between the individual items read when each processing
            each collection of blocks.

        Send:
        ----
        A list of memory views should be sent. They are used to when loading data.

        Return:
        ------
        The generator return value is None.

        """
        # stores eventual returned entries
        flat_pull: List[_T] = []
        # obtain first set of buffers
        buffers: List[memoryview] = yield  # type: ignore
        # go over lists of groups of keys
        for supergroup in batched(self.key_groups(shuffle=shuffle), self.read_size):
            assert len(buffers) >= len(supergroup)
            # get data using buffers
            pulls = [
                self.backing.fused_get(g, buffer=b, presorted=True)
                for g, b in zip(supergroup, buffers)
            ]
            # flatten, shuffle, and yield entries
            flat_pull = list(chain.from_iterable(pulls))
            if shuffle:
                rshuffle(flat_pull)
            # obtain next set of buffers
            buffers = yield flat_pull
        # note that termination happens when one more than needed buffer is provided.
        # in the current usage this does not seem to cause problems.

    def stream(self) -> Iterator[_T]:
        """Serve single data entries via generator.

        Note that entries that are created using this method should be used
        as they are served if the deserialization method uses a view to the original
        buffer; this is because that the memory underlying them will be reused for
        later entries. The exact number of entries that may be drawn before reuse
        is at least as large as self.batch_size.

        Returns:
        -------
        A generator that returns data instances as you iterate.

        Notes:
        -----
        Currently, we do not now allow this method to be called while it is in
        progress.  However, it seems likely that it would be reentrant if we
        simply used a larger set of buffers. If this functionality is needed,
        this analysis can be performed more carefully.

        """
        if self.entered:
            raise ValueError(
                "stream call has seems to have already been entered without exit."
                "This method is not reentrant; if you are sure that the previous "
                "iteration has ended, self the entered attribute to False."
            )
        self.entered = True
        # in a reentrant context, this should create a new _server generator
        serv = self._server(shuffle=self.shuffle)
        # prime generator, mypy does not handle types perfectly
        serv.send(None)  # type: ignore
        # continuously send buffers until we hit a StopIteration.
        # this is how buffer reuse is avoided or delayed; if self._buffers is larger
        # it takes more iterations until we overwrite a given buffer.
        for buffer_selection in cycle(self._buffers):
            try:
                yield from serv.send(buffer_selection)
            except StopIteration:
                # this raises a StopIteration and not RuntimeError, likely due to
                # yield from.
                break
        self.entered = False

    def __iter__(self) -> Iterator[List[_T]]:
        """Serve batches of entries that together cover the entire dataset.

        Batches have a length of self.batch_size, except for a possibly smaller
        final batch.

        Batches should be consumed as they are obtained during iteration. After
        the next batch is served, previous batches may become invalid due to the
        reuse of memory buffers.

        """
        # batched does not use yield from, and so will not pass exceptions from
        # the underlying generator routines. However, it should create its own
        # StopIteration signal and function correctly here.
        yield from batched(self.stream(), self.batch_size)
