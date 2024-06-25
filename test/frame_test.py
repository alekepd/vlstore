"""Test flatbuffer record serialization."""
from typing import Dict
from pytest import mark
import numpy as np
from numpy.random import randint
from random import choice, shuffle
import blosc2  # type: ignore
from vlstore.serialize import BackedAtomicDataCodec, BackedAtomicData
from vlstore.store import Depot, SChunkStore, _create_default_schunk


@mark.parametrize("typesize", [1, 2, 4, 8, 16, 32, 64])
def test_backed_atomic_data_single_groundtruth(typesize: int) -> None:
    """Test serialization of BackedAtomicData on a single molecule type.

    BackedAtomicData instances are extracted and compared to source numpy arrays.
    Multiple typesizes are considered.
    """
    N_ITER = 300
    NAME = "frame_"
    NSITES = 1200
    TYPES = np.arange(NSITES, dtype=np.int32)
    POSITIONS = np.random.rand(N_ITER, NSITES, 3).astype(np.float32)  # noqa: NPY002
    FORCES = np.random.rand(N_ITER, NSITES, 3).astype(np.float32)  # noqa: NPY002
    MASSES = np.random.rand(NSITES).astype(np.float32)  # noqa: NPY002

    cparams = blosc2.cparams_dflts.copy()
    cparams["typesize"] = typesize

    s = _create_default_schunk(chunksize=int(2**22), cparams=cparams)
    storage = SChunkStore(location=s, alignment="no_cross")
    with Depot(codec=BackedAtomicDataCodec, backing=storage) as d:
        for ident, (pos_frame, force_frame) in enumerate(zip(POSITIONS, FORCES)):
            name = NAME + str(ident)
            new = BackedAtomicData.from_values(
                name=NAME,
                masses=MASSES,
                atom_types=TYPES,
                nsites=NSITES,
                positions=pos_frame,
                forces=force_frame,
            )

            d.put(name, new)

    with Depot(codec=BackedAtomicDataCodec, backing=storage) as d:
        for ident, (pos_frame, force_frame) in enumerate(zip(POSITIONS, FORCES)):
            name = NAME + str(ident)
            recreated = d.get(name)
            assert recreated.name == NAME
            assert (recreated.atom_types == TYPES).all()
            assert (recreated.masses == MASSES).all()
            assert (recreated.pos == pos_frame).all()
            assert (recreated.forces == force_frame).all()


def test_backed_atomic_data_multi() -> None:
    """Test serialization of BackedAtomicData for multiple molecule types.

    BackedAtomicData instances are extracted and compared to cached BackedAtomicData
    instances.
    """
    N_ITER = 300
    NAME = "frame_"
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

    storage = SChunkStore(chunksize=int(2**22), location=None, alignment="no_cross")
    record: Dict[str, BackedAtomicData] = {}
    with Depot(codec=BackedAtomicDataCodec, backing=storage) as d:
        for ident in range(N_ITER):
            k = choice(list(NSITES.keys()))
            pos_frame = np.random.rand(NSITES[k], 3).astype(np.float32)  # noqa: NPY002
            force_frame = np.random.rand(NSITES[k], 3).astype(  # noqa: NPY002
                np.float32
            )
            name = NAME + "_" + k + "_" + str(ident)
            new = BackedAtomicData.from_values(
                name=NAME,
                masses=MASSES[k],
                atom_types=TYPES[k],
                nsites=NSITES[k],
                positions=pos_frame,
                forces=force_frame,
            )

            record[name] = new
            d.put(name, new)

    with Depot(codec=BackedAtomicDataCodec, backing=storage) as d:
        keys = list(record.keys())
        shuffle(keys)
        for key in keys:
            stored = record[key]
            recreated = d.get(key)
            assert recreated.name == stored.name
            assert recreated.n_sites == stored.n_sites
            assert (recreated.masses == stored.masses).all()
            assert (recreated.atom_types == stored.atom_types).all()
            assert (recreated.pos == stored.pos).all()
            assert (recreated.forces == stored.forces).all()
