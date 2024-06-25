# vlstore

Fast tools for saving variable-length records on disk.

#### This project is in active development. Interfaces and formats may change without warning.

Data-based applications often read memory many times from disk. Numerical data is often stored in array format, which requires that each element be the same size. When the underlying data is built of variable-sized parts, this causes friction. `vlstore` provides tools to quickly store and load variable length records from a compressed archive.

This project has not been benchmarked against more established strategies. Before use, you should consider the cost of loading your data from disc using tools like [msgpack](https://github.com/msgpack/msgpack), [h5py](https://github.com/h5py/h5py), or database options.

## Key Features

* Persistent on-disk or in-memory storage with fast transparent compression via [python-blosc2](https://github.com/Blosc/python-blosc2)
* Batched record retrieval for reduced I/O operations
* Zero-copy storage formats through [Flatbuffers](https://github.com/google/flatbuffers)

## How To Use

#### Only python 3.9 and later is supported.

First, install the python module. Dependencies should be automatically handled via `pip`.

```bash
# Clone this repository
$ git clone https://github.com/alekepd/vlstore

# Go into the repository
$ cd vlstore

# Install dependencies
$ pip install .
```

Once installed, you need to create a `Codec` which translates your object to and from a `memoryview` object.
A `Codec` is included for data typical to molecular dynamics trajectories; here is how to use it with randomly generated data:

We first import the required libraries.
```python
import numpy as np
# these data types are how we represent and serialize a frame of MD data.
from vlstore.serialize import BackedAtomicDataCodec, BackedAtomicData
# these classes allow us to save and load data from disk.
from vlstore.store import Depot, SChunkStore
```

We then create some random test data to store.

```python
# number of time snapshots to create
N_FRAMES = 300
# number of atoms in each snapshot
NSITES = 1200
# types of each atom
TYPES = np.arange(NSITES, dtype=np.int32)
# masses of each atom
MASSES = np.random.rand(NSITES).astype(np.float32)  # noqa: NPY002
# positions and forces for entire trajectory that we will store
POSITIONS = np.random.rand(N_FRAMES, NSITES, 3).astype(np.float32)  # noqa: NPY002
FORCES = np.random.rand(N_FRAMES, NSITES, 3).astype(np.float32)  # noqa: NPY002
```

This is the name of the file that will contain our saved data.

```python
FILENAME = "mol.store"
```


Now we store that data on disk.

```python
# SChunkStore allows us to store bytes on disk using blosc2, a high performance i/o library.
#  The options here change details of how the data is stored on disk and change performance.
#
#  To create an in-memory compressed store, set FILENAME to None.
storage_interface = SChunkStore(chunksize=int(2**22), location=FILENAME, start_aligned=False)
# Depot allows us to translate our objects to and from bytes, allowing us to store them in the storage_interface.
with Depot(codec=BackedAtomicDataCodec, backing=storage_interface) as d:
    # go through our generated data and store it frame by frame. This is currently the slowest part of the library.
    for ident, (pos_frame, force_frame) in enumerate(zip(POSITIONS, FORCES)):
        name = "frame_" + str(ident)
        # new contains our formatted in-memory data. Doing an operation like this is often required,
        # as fast serialization requires us to understand the stored types in a fundamental way.
        new = BackedAtomicData.from_values(
            name=NAME,
            masses=MASSES,
            atom_types=TYPES,
            nsites=NSITES,
            positions=pos_frame,
            forces=force_frame,
        )

        # store data on disk
        d.put(name, new)

del storage # our data was flushed to disk after leaving the with block
```

Our data is now saved. We can retrieve it with similar code:

```python
# we recreate the interface fresh. When loading existing files, parameters are not needed.
storage_interface = SChunkStore(location=FILENAME)
with Depot(codec=BackedAtomicDataCodec, backing=storage_interface) as d:
    # load single frame
    name = "frame_" + str(0)
    # retrieved is a reconstructed BackedAtomicData object
    retrieved = d.get(name)
```


## License

Apache
