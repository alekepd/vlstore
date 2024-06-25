"""Implements mechanisms for saving and loading data from memory or disk."""
from .chunkstore import (
    SChunkStore,  # noqa: F401
    _create_default_schunk,  # noqa: F401
    DEFAULT_CHUNK_SIZE,  # noqa: F401
    NOCROSS_ALIGNMENT,  # noqa: F401
    CONTIGUOUS_ALIGNMENT,  # noqa: F401
    START_ALIGNMENT,  # noqa: F401
    TYPE_ALIGNMENT,  # noqa: F401
)
from .depot import Depot  # noqa: F401
