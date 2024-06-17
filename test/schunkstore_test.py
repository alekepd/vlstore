"""Tests SChunkStore reads and writes."""
from typing import Final, List, Literal
from pathlib import Path
from random import randbytes, randint, uniform, shuffle
from pytest import mark
from vlstore import SChunkStore
from vlstore.store import DEFAULT_CHUNK_SIZE, _create_default_schunk

# size of writes in bytes
SIZE_SMALL: Final = 500
SIZE_MEDIUM: Final = int(2**13)
SIZE_LARGE: Final = int(2**16)

MANY_SIZE: Final = 100


def test_schunk_small_single() -> None:
    """Test storing a single data point."""
    NAME: Final = "test_name"
    storage = SChunkStore()

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


def test_schunk_small_many() -> None:
    """Test storing many small files."""
    NAME: Final = "test_name"
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_SMALL))
    storage = SChunkStore()
    for name, data in zip(names, content):
        storage.put(name, data)
    for name, data in zip(names, content):
        assert storage.get(name) == data
    for name, data in zip(names, content):
        assert storage[name] == data


def test_schunk_large_many_fused() -> None:
    """Test storing many large files and retrieving them using a fused get.

    All stored items are retrieved and compared.
    """
    NAME: Final = "test_name"
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_LARGE))
    storage = SChunkStore()
    for name, data in zip(names, content):
        storage.put(name, data)

    fused_data, slices = storage.fused_get(names)
    view = memoryview(fused_data)
    views = (view[s] for s in slices)
    for recovered, orig in zip(views, content):
        assert orig == recovered


def test_schunk_large_many_fused_subset() -> None:
    """Test storing many large files and retrieving them using a fused get.

    Every other stored item is retrieved and compared.
    """
    NAME: Final = "test_name"
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_LARGE))
    storage = SChunkStore()
    for name, data in zip(names, content):
        storage.put(name, data)

    subset_slice = slice(None, None, 2)

    fused_data, slices = storage.fused_get(names[subset_slice])
    view = memoryview(fused_data)
    views = (view[s] for s in slices)
    for recovered, orig in zip(views, content[subset_slice]):
        assert orig == recovered


def test_schunk_large_many_fused_subset_shuffle() -> None:
    """Test storing many large files and retrieving them using a fused get.

    Every other stored item is retrieved and compared in a shuffled order.
    """
    NAME: Final = "test_name"
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_LARGE))
    storage = SChunkStore()
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


def test_schunk_large_many() -> None:
    """Test storing many large files."""
    NAME: Final = "test_name"
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_LARGE))
    storage = SChunkStore()
    for name, data in zip(names, content):
        storage.put(name, data)
    for name, data in zip(names, content):
        assert storage.get(name) == data
    for name, data in zip(names, content):
        assert storage[name] == data


def test_schunk_large_many_out() -> None:
    """Test storing many large files using out argument."""
    NAME: Final = "test_name"
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_LARGE))
    storage = SChunkStore()
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
    """Test to see if using random sizes and removing items causes issues."""
    NAME: Final = "test_name"
    DROP_P: Final = 0.1
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_LARGE + randint(1, SIZE_MEDIUM)))

    storage = SChunkStore()
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
    "chunksize",
    [
        100,
        200,
        500,
        int(1e5),
        int(1e8),
        DEFAULT_CHUNK_SIZE,
        DEFAULT_CHUNK_SIZE - 10,
        DEFAULT_CHUNK_SIZE + 10,
        2 * DEFAULT_CHUNK_SIZE,
        SIZE_LARGE,
        SIZE_LARGE + 1,
        SIZE_LARGE - 1,
    ],
)
def test_schunk_large_many_chunksize(chunksize: int) -> None:
    """Test storing many large files."""
    NAME: Final = "test_name"
    names: List[str] = []
    content: List[bytes] = []
    for x in range(MANY_SIZE):
        names.append(NAME + str(x))
        content.append(randbytes(SIZE_LARGE))
    s = _create_default_schunk(chunksize=chunksize)
    storage = SChunkStore(location=s)
    for name, data in zip(names, content):
        storage.put(name, data)
    for name, data in zip(names, content):
        assert storage.get(name) == data
    for name, data in zip(names, content):
        assert storage[name] == data


def test_schunk_manual_numpy() -> None:
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

    storage = SChunkStore()
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


def test_schunk_manual_numpy_disk() -> None:
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
    storage = SChunkStore(location=s)
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
