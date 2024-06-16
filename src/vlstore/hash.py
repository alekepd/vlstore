"""Functions for calculating hashes from data."""
from typing import Final, Any
import xxhash  # type: ignore

INT_MAX_BYTES: Final = 64
INT_BYTE_ORDER: Final = "big"
HASH_SEED: Final = 617491234


def byte_hash(content: Any, /, seed: int = HASH_SEED) -> str:
    """Hash bytes object using xxhash.

    Arguments:
    ---------
    content:
        Array to hash.
    include_shape:
        Whether to make hash depend on each integer in shape tuple.
    seed:
        Initializes hasher.

    Returns:
    -------
    Hash value as an hexadecimal string.

    """
    h = xxhash.xxh3_128(seed=seed)
    h.update(content)
    return h.hexdigest()
