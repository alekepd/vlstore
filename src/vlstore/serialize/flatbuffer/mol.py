"""Scratch script for trying flatbuffers."""
from typing import Final, Any, Optional, Sequence, Union, overload, Literal
from dataclasses import dataclass
import flatbuffers  # type: ignore
from flatbuffers import number_types as N
import numpy as np
from .FBSites import FBFrame
from ...store.util import bytewise_memoryview
from .._types import Codec, T_OUT, TYPE_OUT, TYPE_RETURNDATA, TYPE_INPUTDATA

INI_SIZE: Final = int(2**19)
_builder = flatbuffers.Builder(INI_SIZE)


@dataclass(frozen=True)
class BackedAtomicData:
    """Friendly view into FBFrame.

    Based on atomic data.
    github.com/ClementiGroup/mlcg-tools/blob/
    bc895b0916ca59fb62361be1a27a700816450c2e/mlcg/
    data/atomic_data.py#L22
    """

    raw: memoryview
    FB: FBFrame.FBFrame

    @property
    def name(self) -> int:
        """Name of frame."""
        return self.FB.Name()

    @property
    def n_sites(self) -> int:
        """Number of sites in frame."""
        return self.FB.Nsites()

    @property
    def masses(self) -> np.ndarray:
        """Types of sites in frame."""
        return self.FB.MassesAsNumpy()

    @property
    def atom_types(self) -> np.ndarray:
        """Types of sites in frame."""
        return self.FB.TypesAsNumpy()

    @property
    def pos(self) -> np.ndarray:
        """Portions in frame.

        shape:
        (n_atoms * n_structures=1, 3)
        """
        data = self.FB.PositionsAsNumpy()
        ncodim = self.FB.Ncodim()
        return data.reshape((self.n_sites, ncodim))

    @property
    def forces(self) -> np.ndarray:
        """Forces in frame.

        shape:
        (n_atoms * n_structures=1, 3)
        """
        data = self.FB.ForcesAsNumpy()
        ncodim = self.FB.Ncodim()
        return data.reshape((self.n_sites, ncodim))

    @classmethod
    def from_values(cls, **kwargs) -> "BackedAtomicData":  # noqa
        """Create instance by passing values.

        Values are transformed into an underlying flatbuffer object and used
        to create an instance.

        (Likely arguments, see create_fbframe_buffer for details)

        Arguments:
        ---------
        name: str
            Specifies name of frame (e.g., the name of the protein)
        types: np.ndarray
            Numerical types in system (e.g., element types). Should be int32,
            one dimensional.
        masses: np.ndarray
            Numerical masses in system (e.g., element types). Should be float32.
        nsites: int
            Number of atoms in system.
        positions: np.ndarray
        forces: np.ndarray
        builder: Optional[flatbuffers.Builder] = None

        Returns:
        -------
        Instance of BackedAtomicData

        """
        raw = bytewise_memoryview(create_fbframe_buffer(**kwargs))
        fb = FBFrame.FBFrame.GetRootAs(raw, 0)
        return cls(raw, fb)


@overload
def _pack_BackedAtomicData(
    target: BackedAtomicData,
    /,
    *,
    out: Literal[None] = ...,
) -> TYPE_INPUTDATA:
    ...


@overload
def _pack_BackedAtomicData(
    target: BackedAtomicData,
    /,
    *,
    out: T_OUT,
) -> T_OUT:
    ...


def _pack_BackedAtomicData(
    target: BackedAtomicData,
    out: Optional[TYPE_OUT] = None,
) -> Union[TYPE_INPUTDATA, TYPE_OUT]:
    data = target.raw
    if out is None:
        return data.toreadonly()
    else:
        view = bytewise_memoryview(out)
        view[: len(data)] = data
    return out


def _unpack_BackedAtomicData(data: TYPE_RETURNDATA) -> BackedAtomicData:
    view = bytewise_memoryview(data)
    return BackedAtomicData(view, FBFrame.FBFrame.GetRootAs(view, 0))


BackedAtomicDataCodec = Codec(
    pack=_pack_BackedAtomicData, unpack=_unpack_BackedAtomicData
)


def create_fbframe_buffer(
    name: str,
    atom_types: Sequence[int],
    masses: Sequence[float],
    nsites: int,
    positions: np.ndarray,
    forces: np.ndarray,
    builder: Optional[flatbuffers.Builder] = None,
    check: bool = True,
) -> bytes:
    """Create a bytes object containing FBFrame."""
    # should validate argument type/shape

    if builder is None:
        builder = _builder

    if check:
        if len(forces.shape) != 2:
            raise ValueError()
        if len(positions.shape) != 2:
            raise ValueError()
        if positions.shape[0] != len(masses):
            raise ValueError()
        if positions.shape[0] != len(atom_types):
            raise ValueError()
        if positions.shape != forces.shape:
            raise ValueError()

    nsites = forces.shape[0]

    type_offset = add_types_vector(builder, data=atom_types, fbclass=FBFrame)

    type_offset = add_masses_vector(builder, data=masses, fbclass=FBFrame)

    positions_offset = add_positions_vector(
        builder, data=positions.flatten().tolist(), fbclass=FBFrame
    )

    forces_offset = add_forces_vector(
        builder, data=forces.flatten().tolist(), fbclass=FBFrame
    )

    name_string = builder.CreateString(name)

    FBFrame.Start(builder)

    FBFrame.AddName(builder, name_string)
    FBFrame.AddNsites(builder, nsites)
    FBFrame.AddTypes(builder, type_offset)
    FBFrame.AddMasses(builder, type_offset)
    FBFrame.AddPositions(builder, positions_offset)
    FBFrame.AddForces(builder, forces_offset)

    frame_offset = FBFrame.End(builder)

    builder.Finish(frame_offset)

    content = builder.Output()

    builder.Clear()
    return content


def add_positions_vector(
    builder: flatbuffers.Builder, data: np.ndarray, fbclass: Any
) -> int:
    """Add positions (float32) vector."""
    fbclass.StartPositionsVector(builder, len(data))
    reversed_data = reversed(data)
    # typically, the call should use: _call = builder.PrependFloat32
    # we can get minor speed increases by inlining the corresponding operations,
    # but this comes at a risk.
    _call = builder.Prepend

    # can probably shave off some lookups
    # https://github.com/google/flatbuffers/blob/dafd2f1f29c5c0c2a6df285e9efe5c49746c6b7a
    # /python/flatbuffers/builder.py#L638

    dtype = N.Float32Flags
    for i in reversed_data:
        _call(dtype, i)
    return builder.EndVector()


def add_forces_vector(
    builder: flatbuffers.Builder, data: np.ndarray, fbclass: Any
) -> int:
    """Add forces (float32) vector."""
    fbclass.StartForcesVector(builder, len(data))
    reversed_data = reversed(data)
    # typically, the call should use: _call = builder.PrependFloat32
    # we can get minor speed increases by inlining the corresponding operations,
    # but this comes at a risk.
    _call = builder.Prepend

    dtype = N.Float32Flags
    for i in reversed_data:
        _call(dtype, i)
    return builder.EndVector()


def add_masses_vector(
    builder: flatbuffers.Builder, data: Sequence[float], fbclass: Any
) -> int:
    """Add mass (float32) vector."""
    fbclass.StartMassesVector(builder, len(data))
    reversed_data = reversed(data)
    # typically, the call should use: _call = builder.PrependFloat32
    # we can get minor speed increases by inlining the corresponding operations,
    # but this comes at a risk.
    _call = builder.Prepend

    dtype = N.Float32Flags
    for i in reversed_data:
        _call(dtype, i)
    return builder.EndVector()


def add_types_vector(
    builder: flatbuffers.Builder, data: Sequence[int], fbclass: Any
) -> int:
    """Add int32 vector."""
    fbclass.StartTypesVector(builder, len(data))
    reversed_data = reversed(data)
    # see corresponding float32 version: _call = builder.PrependInt32
    _call = builder.Prepend
    dtype = N.Int32Flags
    for i in reversed_data:
        _call(dtype, i)
    return builder.EndVector()
