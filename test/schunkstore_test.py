"""Tests SChunkStore reads and writes."""
from typing import Final, List, Literal
from pathlib import Path
from random import randbytes, randint, uniform, shuffle
from pytest import mark
import blosc2  # type: ignore
from vlstore import SChunkStore
from vlstore.store import (
    DEFAULT_CHUNK_SIZE,
    TYPE_ALIGNMENT,
    NOCROSS_ALIGNMENT,
    CONTIGUOUS_ALIGNMENT,
    START_ALIGNMENT,
    _create_default_schunk,
)

# size of writes in bytes
SIZE_SMALL: Final = 500
SIZE_MEDIUM: Final = int(2**13)
SIZE_LARGE: Final = int(2**16)

MANY_SIZE: Final = 100


@mark.parametrize(
    "alignment",
    [
        NOCROSS_ALIGNMENT,
        CONTIGUOUS_ALIGNMENT,
        START_ALIGNMENT,
    ],
)
def test_schunk_small_single(alignment: TYPE_ALIGNMENT) -> None:
    """Test storing a single data point."""
    NAME: Final = "test_name"
    storage = SChunkStore(alignment=alignment)

    content = randbytes(SIZE_SMALL)
    storage.put(NAME, content)
    reproduced_content = storage.get(NAME)
    assert reproduced_content == content
    reproduced_content = storage[NAME]
    assert reproduced_content == content

    content = randbytes(SIZE_SMALL)
    storage[NAME] = content
    reproduced_content = storage.get(NAME)
    assert reproduced_content == content
    reproduced_content = storage[NAME]
    assert reproduced_content == content

    try:
        content = randbytes(SIZE_SMALL)
        storage.put(NAME, value=content, overwrite=False)
    except ValueError:
        pass
    else:
        raise AssertionError()


@mark.parametrize(
    "alignment",
    [
        NOCROSS_ALIGNMENT,
        CONTIGUOUS_ALIGNMENT,
        START_ALIGNMENT,
    ],
)
def test_schunk_small_many(alignment: TYPE_ALIGNMENT) -> None:
    """Test storing many small files."""
    NAME: Final = "test_name"
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_SMALL))
    storage = SChunkStore(alignment=alignment)
    for name, data in zip(names, content):
        storage.put(name, data)
    for name, data in zip(names, content):
        assert storage.get(name) == data
    for name, data in zip(names, content):
        assert storage[name] == data


@mark.parametrize(
    "alignment",
    [
        NOCROSS_ALIGNMENT,
        CONTIGUOUS_ALIGNMENT,
        START_ALIGNMENT,
    ],
)
def test_schunk_large_many_fused(alignment: TYPE_ALIGNMENT) -> None:
    """Test storing many large files and retrieving them using a fused get.

    All stored items are retrieved and compared.
    """
    NAME: Final = "test_name"
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_LARGE))
    storage = SChunkStore(alignment=alignment)
    for name, data in zip(names, content):
        storage.put(name, data)

    fused_data, slices = storage.fused_get(names)
    view = memoryview(fused_data)
    views = (view[s] for s in slices)
    for recovered, orig in zip(views, content):
        assert orig == recovered


@mark.parametrize(
    "alignment",
    [
        NOCROSS_ALIGNMENT,
        CONTIGUOUS_ALIGNMENT,
        START_ALIGNMENT,
    ],
)
def test_schunk_large_many_fused_subset(alignment: TYPE_ALIGNMENT) -> None:
    """Test storing many large files and retrieving them using a fused get.

    Every other stored item is retrieved and compared.
    """
    NAME: Final = "test_name"
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_LARGE))
    storage = SChunkStore(alignment=alignment)
    for name, data in zip(names, content):
        storage.put(name, data)

    subset_slice = slice(None, None, 2)

    fused_data, slices = storage.fused_get(names[subset_slice])
    view = memoryview(fused_data)
    views = (view[s] for s in slices)
    for recovered, orig in zip(views, content[subset_slice]):
        assert orig == recovered


@mark.parametrize(
    "alignment",
    [
        NOCROSS_ALIGNMENT,
        CONTIGUOUS_ALIGNMENT,
        START_ALIGNMENT,
    ],
)
def test_schunk_large_many_fused_subset_shuffle(alignment: TYPE_ALIGNMENT) -> None:
    """Test storing many large files and retrieving them using a fused get.

    Every other stored item is retrieved and compared in a shuffled order.
    """
    NAME: Final = "test_name"
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_LARGE))
    storage = SChunkStore(alignment=alignment)
    data_pairing = dict(zip(names, content))
    for name, data in data_pairing.items():
        storage.put(name, data)

    subset_slice = slice(None, None, 2)
    subnames = names[subset_slice]
    shuffle(subnames)

    fused_data, slices = storage.fused_get(subnames)
    view = memoryview(fused_data)
    views = (view[s] for s in slices)
    for recovered, key in zip(views, subnames):
        assert data_pairing[key] == recovered


@mark.parametrize(
    "alignment",
    [
        NOCROSS_ALIGNMENT,
        CONTIGUOUS_ALIGNMENT,
        START_ALIGNMENT,
    ],
)
def test_schunk_large_many(alignment: TYPE_ALIGNMENT) -> None:
    """Test storing many large files."""
    NAME: Final = "test_name"
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_LARGE))
    storage = SChunkStore(alignment=alignment)
    for name, data in zip(names, content):
        storage.put(name, data)
    for name, data in zip(names, content):
        assert storage.get(name) == data
    for name, data in zip(names, content):
        assert storage[name] == data


@mark.parametrize(
    "alignment",
    [
        NOCROSS_ALIGNMENT,
        CONTIGUOUS_ALIGNMENT,
        START_ALIGNMENT,
    ],
)
def test_schunk_large_many_out(alignment: TYPE_ALIGNMENT) -> None:
    """Test storing many large files using out argument."""
    NAME: Final = "test_name"
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_LARGE))
    storage = SChunkStore(alignment=alignment)
    for name, data in zip(names, content):
        storage.put(name, data)
    for name, data in zip(names, content):
        b = bytearray(len(data))
        assert storage.get(name, out=b) == data


@mark.parametrize(
    "get_method",
    [
        "slice",
        "chunk",
    ],
)
def test_schunk_random_many_get(get_method: Literal["slice", "chunk"]) -> None:
    """Test to see if using random sizes and removing items causes issues.

    Allocation is force to be aligned to allow multiple get methods.
    """
    NAME: Final = "test_name"
    DROP_P: Final = 0.1
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_LARGE + randint(1, SIZE_MEDIUM)))

    storage = SChunkStore(alignment=START_ALIGNMENT)
    for name, data in zip(names, content):
        storage.put(name, data)

    for name, data in zip(names, content):
        if uniform(0, 1) < DROP_P:
            names.remove(name)
            content.remove(data)
            storage.disown(name)
            assert name not in storage.lookup

    for name, data in zip(names, content):
        assert storage.get(name, method=get_method) == data
    for name, data in zip(names, content):
        assert storage[name] == data


@mark.parametrize(
    "chunksize,alignment",
    [
        (100, NOCROSS_ALIGNMENT),
        (200, NOCROSS_ALIGNMENT),
        (500, NOCROSS_ALIGNMENT),
        (int(1e5), NOCROSS_ALIGNMENT),
        (int(1e8), NOCROSS_ALIGNMENT),
        (DEFAULT_CHUNK_SIZE, NOCROSS_ALIGNMENT),
        (DEFAULT_CHUNK_SIZE - 10, NOCROSS_ALIGNMENT),
        (DEFAULT_CHUNK_SIZE + 10, NOCROSS_ALIGNMENT),
        (2 * DEFAULT_CHUNK_SIZE, NOCROSS_ALIGNMENT),
        (SIZE_LARGE, NOCROSS_ALIGNMENT),
        (SIZE_LARGE + 1, NOCROSS_ALIGNMENT),
        (SIZE_LARGE - 1, NOCROSS_ALIGNMENT),
        (2 * SIZE_LARGE + 10, NOCROSS_ALIGNMENT),
        (100, CONTIGUOUS_ALIGNMENT),
        (200, CONTIGUOUS_ALIGNMENT),
        (500, CONTIGUOUS_ALIGNMENT),
        (int(1e5), CONTIGUOUS_ALIGNMENT),
        (int(1e8), CONTIGUOUS_ALIGNMENT),
        (DEFAULT_CHUNK_SIZE, CONTIGUOUS_ALIGNMENT),
        (DEFAULT_CHUNK_SIZE - 10, CONTIGUOUS_ALIGNMENT),
        (DEFAULT_CHUNK_SIZE + 10, CONTIGUOUS_ALIGNMENT),
        (2 * DEFAULT_CHUNK_SIZE, CONTIGUOUS_ALIGNMENT),
        (SIZE_LARGE, CONTIGUOUS_ALIGNMENT),
        (SIZE_LARGE + 1, CONTIGUOUS_ALIGNMENT),
        (SIZE_LARGE - 1, CONTIGUOUS_ALIGNMENT),
        (2 * SIZE_LARGE + 10, CONTIGUOUS_ALIGNMENT),
        (100, START_ALIGNMENT),
        (200, START_ALIGNMENT),
        (500, START_ALIGNMENT),
        (int(1e5), START_ALIGNMENT),
        (int(1e8), START_ALIGNMENT),
        (DEFAULT_CHUNK_SIZE, START_ALIGNMENT),
        (DEFAULT_CHUNK_SIZE - 10, START_ALIGNMENT),
        (DEFAULT_CHUNK_SIZE + 10, START_ALIGNMENT),
        (2 * DEFAULT_CHUNK_SIZE, START_ALIGNMENT),
        (SIZE_LARGE, START_ALIGNMENT),
        (SIZE_LARGE + 1, START_ALIGNMENT),
        (SIZE_LARGE - 1, START_ALIGNMENT),
        (2 * SIZE_LARGE + 10, START_ALIGNMENT),
    ],
)
def test_schunk_large_many_chunksize(chunksize: int, alignment: TYPE_ALIGNMENT) -> None:
    """Test storing many large files with different chunk sizes."""
    NAME: Final = "test_name"
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_LARGE))
    s = _create_default_schunk(chunksize=chunksize)
    storage = SChunkStore(location=s, alignment=alignment)
    for name, data in zip(names, content):
        storage.put(name, data)
    for name, data in zip(names, content):
        assert storage.get(name) == data
    for name, data in zip(names, content):
        assert storage[name] == data


@mark.parametrize(
    "chunksize,alignment,typesize",
    [
        (int(2**10), NOCROSS_ALIGNMENT, 1),
        (int(2**10), NOCROSS_ALIGNMENT, 2),
        (int(2**10), NOCROSS_ALIGNMENT, 4),
        (int(2**10), NOCROSS_ALIGNMENT, 8),
        (int(2**15), NOCROSS_ALIGNMENT, 1),
        (int(2**15), NOCROSS_ALIGNMENT, 2),
        (int(2**15), NOCROSS_ALIGNMENT, 4),
        (int(2**15), NOCROSS_ALIGNMENT, 8),
        (int(2**20), NOCROSS_ALIGNMENT, 1),
        (int(2**20), NOCROSS_ALIGNMENT, 2),
        (int(2**20), NOCROSS_ALIGNMENT, 4),
        (int(2**20), NOCROSS_ALIGNMENT, 8),
        (int(2**10), CONTIGUOUS_ALIGNMENT, 1),
        (int(2**10), CONTIGUOUS_ALIGNMENT, 2),
        (int(2**10), CONTIGUOUS_ALIGNMENT, 4),
        (int(2**10), CONTIGUOUS_ALIGNMENT, 8),
        (int(2**15), CONTIGUOUS_ALIGNMENT, 1),
        (int(2**15), CONTIGUOUS_ALIGNMENT, 2),
        (int(2**15), CONTIGUOUS_ALIGNMENT, 4),
        (int(2**15), CONTIGUOUS_ALIGNMENT, 8),
        (int(2**20), CONTIGUOUS_ALIGNMENT, 1),
        (int(2**20), CONTIGUOUS_ALIGNMENT, 2),
        (int(2**20), CONTIGUOUS_ALIGNMENT, 4),
        (int(2**20), CONTIGUOUS_ALIGNMENT, 8),
        (int(2**10), START_ALIGNMENT, 1),
        (int(2**10), START_ALIGNMENT, 2),
        (int(2**10), START_ALIGNMENT, 4),
        (int(2**10), START_ALIGNMENT, 8),
        (int(2**15), START_ALIGNMENT, 1),
        (int(2**15), START_ALIGNMENT, 2),
        (int(2**15), START_ALIGNMENT, 4),
        (int(2**15), START_ALIGNMENT, 8),
        (int(2**20), START_ALIGNMENT, 1),
        (int(2**20), START_ALIGNMENT, 2),
        (int(2**20), START_ALIGNMENT, 4),
        (int(2**20), START_ALIGNMENT, 8),
    ],
)
def test_schunk_large_many_typesize(
    chunksize: int, alignment: TYPE_ALIGNMENT, typesize: int
) -> None:
    """Test storing many large files with different type and chunk sizes."""
    NAME: Final = "test_name"
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_LARGE))

    cparams = blosc2.cparams_dflts.copy()
    cparams["typesize"] = typesize
    cparams["codec"] = blosc2.Codec.LZ4HC

    s = _create_default_schunk(chunksize=chunksize, cparams=cparams)
    storage = SChunkStore(location=s, alignment=alignment)
    for name, data in zip(names, content):
        storage.put(name, data)
    for name, data in zip(names, content):
        assert storage.get(name) == data
    for name, data in zip(names, content):
        assert storage[name] == data


@mark.parametrize(
    "alignment",
    [
        NOCROSS_ALIGNMENT,
        CONTIGUOUS_ALIGNMENT,
        START_ALIGNMENT,
    ],
)
def test_schunk_manual_numpy(alignment: TYPE_ALIGNMENT) -> None:
    """Test if numpy arrays can be manually serialized."""
    from numpy.random import default_rng
    import numpy as np

    NAME_1: Final = "array_1"
    NAME_2: Final = "array_2"
    rng = default_rng()
    SHAPE_1: Final = (3, 5, 6)
    data1 = rng.random(SHAPE_1, dtype=np.float32)
    SHAPE_2: Final = (62, 450, 53, 37)
    data2 = rng.random(SHAPE_2, dtype=np.float64)

    storage = SChunkStore(alignment=alignment)
    storage.put(NAME_1, data1.tobytes())
    storage.put(NAME_2, data2.tobytes())

    extracted_buffer_1 = storage.get(NAME_1)
    extracted_buffer_2 = storage.get(NAME_2)

    recreated1 = np.frombuffer(extracted_buffer_1, dtype=np.float32)
    recreated2 = np.frombuffer(extracted_buffer_2, dtype=np.float64)

    assert (recreated1.reshape(data1.shape) == data1).all()
    assert (recreated2.reshape(data2.shape) == data2).all()

    premade1 = np.zeros_like(data1)
    storage.get(NAME_1, out=premade1)
    premade2 = np.zeros_like(data2)
    storage.get(NAME_2, out=premade2)

    assert (premade1 == data1).all()
    assert (premade2 == data2).all()


@mark.parametrize(
    "alignment",
    [
        NOCROSS_ALIGNMENT,
        CONTIGUOUS_ALIGNMENT,
        START_ALIGNMENT,
    ],
)
def test_schunk_manual_numpy_disk(alignment: TYPE_ALIGNMENT) -> None:
    """Test if numpy arrays can be manually serialized.

    Uses on disk storage for underlying schunk.
    """
    from numpy.random import default_rng
    import numpy as np

    FILENAME: Final = Path("storage.schunk2")
    NAME_1: Final = "array_1"
    NAME_2: Final = "array_2"
    rng = default_rng()
    SHAPE_1: Final = (3, 5, 6)
    data1 = rng.random(SHAPE_1, dtype=np.float32)
    SHAPE_2: Final = (62, 450, 53, 37)
    data2 = rng.random(SHAPE_2, dtype=np.float64)

    s = _create_default_schunk(filename=FILENAME)
    storage = SChunkStore(location=s, alignment=alignment)
    storage.put(NAME_1, data1.tobytes())
    storage.put(NAME_2, data2.tobytes())

    extracted_buffer_1 = storage.get(NAME_1)
    extracted_buffer_2 = storage.get(NAME_2)

    recreated1 = np.frombuffer(extracted_buffer_1, dtype=np.float32)
    recreated2 = np.frombuffer(extracted_buffer_2, dtype=np.float64)

    assert (recreated1.reshape(data1.shape) == data1).all()
    assert (recreated2.reshape(data2.shape) == data2).all()

    premade1 = np.zeros_like(data1)
    storage.get(NAME_1, out=premade1)
    premade2 = np.zeros_like(data2)
    storage.get(NAME_2, out=premade2)

    assert (premade1 == data1).all()
    assert (premade2 == data2).all()

    FILENAME.unlink()
