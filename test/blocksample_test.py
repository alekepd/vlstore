"""Tests for blocked batch loader."""
from typing import Dict
from pytest import mark
import numpy as np
from numpy.random import randint
from random import choice
from vlstore.serialize import BackedAtomicDataCodec, BackedAtomicData
from vlstore.store import Depot, SChunkStore
from vlstore.loader import BlockShuffleBatch


@mark.parametrize(
    "n_buffers,read_size,end_block_merge,shuffle",
    [
        (2, 3, True, True),
        (2, 3, True, False),
        (2, 3, False, True),
        (2, 3, False, False),
        (2, 4, True, True),
        (2, 4, True, False),
        (2, 4, False, True),
        (2, 4, False, False),
        (2, 5, True, True),
        (2, 5, True, False),
        (2, 5, False, True),
        (2, 5, False, False),
        (3, 5, True, True),
        (3, 5, True, False),
        (3, 5, False, True),
        (3, 5, False, False),
        (4, 5, True, True),
        (4, 5, True, False),
        (4, 5, False, True),
        (4, 5, False, False),
    ],
)
def test_block_stream_noshuffle_atomic_data_multi(
    n_buffers: int, read_size: int, end_block_merge: bool, shuffle: bool
) -> None:
    """Test loader stream method.

    Data is checked for content based on stored atomic data instances.

    In-memory storage is used.
    """
    N_ITER = 2000  # number of frames in test
    NAME = "frame"
    M1, M2, M3 = "MOL_A", "MOL_B", "MOL_C"
    NSITES = {M1: 1200, M2: 882, M3: 100}
    TYPES = {
        x: randint(low=0, high=20, size=NSITES[x], dtype=np.int32)  # noqa: NPY002
        for x in NSITES.keys()
    }
    MASSES = {
        k: np.random.rand(v).astype(np.float32)  # noqa: NPY002
        for k, v in NSITES.items()
    }

    storage = SChunkStore(chunksize=int(2**19), location=None, alignment="no_cross")
    record: Dict[str, BackedAtomicData] = {}
    # create and store data in both depo and local dictionary.
    with Depot(codec=BackedAtomicDataCodec, backing=storage) as d:
        for ident in range(N_ITER):
            k = choice(list(NSITES.keys()))
            pos_frame = np.random.rand(NSITES[k], 3).astype(np.float32)  # noqa: NPY002
            force_frame = np.random.rand(NSITES[k], 3).astype(  # noqa: NPY002
                np.float32
            )
            name = NAME + "_" + k + "_" + str(ident)
            new = BackedAtomicData.from_values(
                name=name,
                masses=MASSES[k],
                atom_types=TYPES[k],
                nsites=NSITES[k],
                positions=pos_frame,
                forces=force_frame,
            )

            record[name] = new
            d.put(name, new)

    # read back data and see if is in the correct order and matches each data instance.
    with Depot(codec=BackedAtomicDataCodec, backing=storage) as d:
        loader = BlockShuffleBatch(
            read_size=read_size,
            batch_size=1,
            backing=d,
            shuffle=shuffle,
            end_block_merge=end_block_merge,
            n_buffers=n_buffers,
        )

        located_keys = set()

        for recreated in loader.stream():
            key = recreated.name
            located_keys.add(key)
            stored = record[recreated.name]
            assert recreated.name == stored.name
            assert recreated.n_sites == stored.n_sites
            assert (recreated.masses == stored.masses).all()
            assert (recreated.atom_types == stored.atom_types).all()
            assert (recreated.pos == stored.pos).all()
            assert (recreated.forces == stored.forces).all()
        stored_keys = set(record.keys())
        assert len(stored_keys - located_keys) == 0


@mark.parametrize(
    "read_size,batch_size,shuffle",
    [
        (2, 2, False),
        (2, 5, False),
        (2, 10, False),
        (2, 2, True),
        (2, 5, True),
        (2, 10, True),
        (5, 2, False),
        (5, 5, False),
        (5, 10, False),
        (5, 2, True),
        (5, 5, True),
        (5, 10, True),
    ],
)
def test_block_batch_atomic_data_multi(
    read_size: int, batch_size: int, shuffle: bool
) -> None:
    """Test loader iteration (batch) method.

    Data is checked for content based on stored atomic data instances.

    In-memory storage is used.
    """
    N_ITER = 2000  # number of frames in test
    NAME = "frame"
    M1, M2, M3 = "MOL_A", "MOL_B", "MOL_C"
    NSITES = {M1: 1200, M2: 882, M3: 100}
    TYPES = {
        x: randint(low=0, high=20, size=NSITES[x], dtype=np.int32)  # noqa: NPY002
        for x in NSITES.keys()
    }
    MASSES = {
        k: np.random.rand(v).astype(np.float32)  # noqa: NPY002
        for k, v in NSITES.items()
    }

    storage = SChunkStore(chunksize=int(2**19), location=None, alignment="no_cross")
    record: Dict[str, BackedAtomicData] = {}
    # create and store data in both depo and local dictionary.
    with Depot(codec=BackedAtomicDataCodec, backing=storage) as d:
        for ident in range(N_ITER):
            k = choice(list(NSITES.keys()))
            pos_frame = np.random.rand(NSITES[k], 3).astype(np.float32)  # noqa: NPY002
            force_frame = np.random.rand(NSITES[k], 3).astype(  # noqa: NPY002
                np.float32
            )
            name = NAME + "_" + k + "_" + str(ident)
            new = BackedAtomicData.from_values(
                name=name,
                masses=MASSES[k],
                atom_types=TYPES[k],
                nsites=NSITES[k],
                positions=pos_frame,
                forces=force_frame,
            )

            record[name] = new
            d.put(name, new)

    # read back data and see if is in the correct order and matches each data instance.
    with Depot(codec=BackedAtomicDataCodec, backing=storage) as d:
        loader = BlockShuffleBatch(
            read_size=read_size, batch_size=batch_size, backing=d, shuffle=shuffle
        )

        located_keys = set()

        for recreated in loader:
            for frame in recreated:
                key = frame.name
                located_keys.add(key)
                stored = record[frame.name]
                assert frame.name == stored.name
                assert frame.n_sites == stored.n_sites
                assert (frame.masses == stored.masses).all()
                assert (frame.atom_types == stored.atom_types).all()
                assert (frame.pos == stored.pos).all()
                assert (frame.forces == stored.forces).all()
        stored_keys = set(record.keys())
        assert len(stored_keys - located_keys) == 0
