"""Tests Depot with numpy serialization reads and writes."""

from typing import Final
from random import shuffle
from pathlib import Path
import numpy as np
from vlstore.serialize import flatfloatndarray_codec
from vlstore.store import SChunkStore, Depot
from vlstore.hash import byte_hash

SIZE: Final = int(2**20)
SIZE_DELTA: Final = int(2**23)

_rng = np.random.default_rng()


def test_array_single() -> None:
    """Tests storing and recovering single array."""
    NAME: Final = "test_name"
    data = _rng.random(SIZE, dtype=np.float32)
    d = Depot(codec=flatfloatndarray_codec)
    d.put(NAME, data)
    recovered = d.get(NAME)
    assert (data == recovered).all()


def test_array_multiple_shuffle_variedsize() -> None:
    """Tests storing and recovering multiple stored arrays.

    Content has varying sizes and is retrieved out of order.
    """
    NAME: Final = "test_name"
    NUM_RECORDS: Final = 50
    d = Depot(codec=flatfloatndarray_codec)
    data_record = {}
    for x in range(NUM_RECORDS):
        name = NAME + str(x)
        size = SIZE + int(np.floor(SIZE_DELTA * _rng.random()))
        data = _rng.random(size, dtype=np.float32)
        data_record[name] = data
        d.put(name, data)
    keys = list(data_record.keys())
    shuffle(keys)
    for key in keys:
        assert (data_record[key] == d.get(key)).all()


def test_array_multiple_shuffle_variedsize_disk_hash() -> None:
    """Tests storing and recovering multiple stored arrays.

    Content has varying sizes and is retrieved out of order. Only some content
    is retrieved.

    Storage is disk-backed.

    Comparison is hash based.
    """
    NAME: Final = "test_name"
    FILENAME: Final = Path("storage.schunk2")
    NUM_RECORDS: Final = 10
    BIGGER_SIZE = 10 * SIZE
    storage = SChunkStore(location=FILENAME)
    d = Depot(codec=flatfloatndarray_codec, backing=storage)
    hash_record = {}
    for x in range(NUM_RECORDS):
        name = NAME + str(x)
        size = BIGGER_SIZE + int(np.floor(SIZE_DELTA * _rng.random()))
        data = _rng.random(size, dtype=np.float32)
        hash_record[name] = byte_hash(data.tobytes())
        d.put(name, data)
    keys = list(hash_record.keys())
    shuffle(keys)
    for key in keys[::2]:
        assert hash_record[key] == byte_hash(d.get(key).tobytes())

    FILENAME.unlink()


def test_array_openclose_hash() -> None:
    """Tests storing and recovering multiple stored arrays after file close and opening.

    No context manager is used; close is called manually on the SChunkStore.

    Content has varying sizes and is retrieved out of order. Only some content
    is retrieved.

    Storage is disk-backed, comparison is hash based.
    """
    NAME: Final = "test_name"
    FILENAME: Final = Path("storage.schunk2")
    NUM_RECORDS: Final = 10
    BIGGER_SIZE = 10 * SIZE
    storage = SChunkStore(location=FILENAME)
    d = Depot(codec=flatfloatndarray_codec, backing=storage)
    hash_record = {}
    for x in range(NUM_RECORDS):
        name = NAME + str(x)
        size = BIGGER_SIZE + int(np.floor(SIZE_DELTA * _rng.random()))
        data = _rng.random(size, dtype=np.float32)
        hash_record[name] = byte_hash(data.tobytes())
        d.put(name, data)
    storage.close()

    del d
    del storage

    keys = list(hash_record.keys())
    shuffle(keys)

    storage = SChunkStore(location=FILENAME)
    d = Depot(codec=flatfloatndarray_codec, backing=storage)

    for key in keys[::2]:
        assert hash_record[key] == byte_hash(d.get(key).tobytes())

    FILENAME.unlink()


def test_array_chunkcontextm_hash() -> None:
    """Tests storing and recovering multiple stored arrays using context manager.

    with construct is used.

    Content has varying sizes and is retrieved out of order. Only some content
    is retrieved.

    Storage is disk-backed, comparison is hash based.
    """
    NAME: Final = "test_name"
    FILENAME: Final = Path("storage.schunk2")
    NUM_RECORDS: Final = 10
    BIGGER_SIZE = 10 * SIZE
    with SChunkStore(location=FILENAME) as storage:
        d = Depot(codec=flatfloatndarray_codec, backing=storage)
        hash_record = {}
        for x in range(NUM_RECORDS):
            name = NAME + str(x)
            size = BIGGER_SIZE + int(np.floor(SIZE_DELTA * _rng.random()))
            data = _rng.random(size, dtype=np.float32)
            hash_record[name] = byte_hash(data.tobytes())
            d.put(name, data)

    del d

    keys = list(hash_record.keys())
    shuffle(keys)

    with SChunkStore(location=FILENAME) as storage:
        d = Depot(codec=flatfloatndarray_codec, backing=storage)

        for key in keys[::2]:
            assert hash_record[key] == byte_hash(d.get(key).tobytes())

    FILENAME.unlink()
