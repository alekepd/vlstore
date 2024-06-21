"""Test flatbuffer record serialization."""
from typing import Final
import numpy as np
from tqdm import tqdm
from vlstore.serialize.flatbuffer.mol import BackedAtomicDataCodec, BackedAtomicData
from vlstore.store import Depot, SChunkStore


def test() -> None:
    """Minimal test for correctness."""
    N_ITER = 20000
    NAME = "frame_"
    NSITES = 1200
    TYPES = np.arange(NSITES, dtype=np.int32)
    POSITIONS = np.random.rand(N_ITER, NSITES, 3).astype(np.float32)  # noqa: NPY002
    FORCES = np.random.rand(N_ITER, NSITES, 3).astype(np.float32)  # noqa: NPY002
    MASSES = np.random.rand(NSITES)  # noqa: NPY002

    FILENAME: Final = "mol.store"
    storage = SChunkStore(chunksize=int(2**21),location=FILENAME, start_aligned=False)
    with Depot(codec=BackedAtomicDataCodec, backing=storage) as d:
        for ident, (pos_frame, force_frame) in tqdm(enumerate(zip(POSITIONS,FORCES))):
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
        print(d.backing.backing.cratio)

    with Depot(codec=BackedAtomicDataCodec, backing=storage) as d:
        for ident, (pos_frame, force_frame) in tqdm(enumerate(zip(POSITIONS,FORCES))):
            name = NAME + str(ident)
            recreated = d.get(name)
            assert (recreated.pos == pos_frame).all()
            assert (recreated.forces == force_frame).all()
