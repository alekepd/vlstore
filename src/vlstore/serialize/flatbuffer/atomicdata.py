"""Scratch script for trying flatbuffers."""
from typing import Final, Any, Optional, Sequence, Union, overload, Literal
from dataclasses import dataclass
import flatbuffers  # type: ignore
from flatbuffers import number_types as N
import numpy as np
from .FBSites import FBFrame
from ...store.util import bytewise_memoryview
from .._types import Codec, T_OUT, TYPE_OUT, TYPE_RETURNDATA, TYPE_INPUTDATA

# Specifies default starting size for builder
INI_SIZE: Final = int(2**19)
# buffer object that is used to create flatbuffer objects if none is given
_builder = flatbuffers.Builder(INI_SIZE)


@dataclass(frozen=True)
class BackedAtomicData:
    """Frame of particle data.

    This object is backed by an FBFrame instance; most attributes are properties
    which access and transform the underlying data. It is also the most convenient
    way to create an FBFrame instance.

    Note that __init__ takes an already formed FBFrame instance; for easy creation,
    see alternative init .from_values.

    Attributes:
    ----------
    raw:
        Buffer containing memory used by underlying FBFrame.
    fb:
        Underlying FBFrame instance.
    name:
        String identifying record.
    n_sites:
        Number of sites or particles in frame.
    masses:
        Masses present in system; numpy float32 array.
    atom_types:
        Masses present in system; numpy float32 array.
    pos:
        Positions of frames in system; numpy float32 array of
        shape (n_particles,n_codims). Codims is almost always 3.
    forces
        Forces of frames in system; numpy float32 array of
        shape (n_particles,n_codims). Codims is almost always 3.

    """

    raw: memoryview
    fb: FBFrame.FBFrame

    @property
    def name(self) -> int:
        """Name of frame."""
        return self.fb.Name()

    @property
    def n_sites(self) -> int:
        """Number of sites in frame."""
        return self.fb.Nsites()

    @property
    def masses(self) -> np.ndarray:
        """Masses of sites in frame.

        float32 numpy.ndarray of shape (self.n_sites,).
        """
        return self.fb.MassesAsNumpy()

    @property
    def atom_types(self) -> np.ndarray:
        """Types of sites in frame.

        float32 numpy.ndarray of shape (self.n_sites,).
        """
        return self.fb.TypesAsNumpy()

    @property
    def pos(self) -> np.ndarray:
        """Positions in frame.

        float32 numpy.ndarray of shape (self.n_sites, codim). Codim is almost
        always 3; see FBFrame for more information.
        """
        data = self.fb.PositionsAsNumpy()
        ncodim = self.fb.Ncodim()
        return data.reshape((self.n_sites, ncodim))

    @property
    def forces(self) -> np.ndarray:
        """Forces in frame.

        float32 numpy.ndarray of shape (self.n_sites, codim). Codim is almost
        always 3; see FBFrame for more information.
        """
        data = self.fb.ForcesAsNumpy()
        ncodim = self.fb.Ncodim()
        return data.reshape((self.n_sites, ncodim))

    @classmethod
    def from_values(cls, **kwargs) -> "BackedAtomicData":  # noqa
        """Create instance by passing defining values.

        Values are transformed into an underlying flatbuffer object and used
        to create an instance.

        (The following are likely arguments, see create_fbframe_buffer for details)

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
    """Serialize BackedAtomicData instance.

    This routing simply serves the buffer underlying target. If out is
    not provided, note that this is effectively a view.
    """
    data = target.raw
    if out is None:
        return data.toreadonly()
    else:
        view = bytewise_memoryview(out)
        view[: len(data)] = data
    return out


def _unpack_BackedAtomicData(data: TYPE_RETURNDATA) -> BackedAtomicData:
    """Create BackedAtomicData from buffer.

    Creates an FBFrame instance and then a BackedAtomicData instance.
    """
    view = bytewise_memoryview(data)
    return BackedAtomicData(view, FBFrame.FBFrame.GetRootAs(view, 0))


# This is used to pack and unpack BackedAtomicData instances.
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
    """Create a bytes object containing FBFrame.

    Note that some of the argument names disagree in formatting relative to
    BackedAtomicData; Consider using BackedAtomicData.from_values instead.

    This method is slow because the FlatBuffer python interface is slow.

    Arguments:
    ---------
    name:
        String identifying record.
    atom_types:
        Types present in frame; Sequence of ints of length nsites.
    masses:
        Masses present in frame; Sequence of floats of length nsites.
    nsites:
        Number of sites present; dictates acceptable shapes of other arguments.
    positions:
        Positions of frames in system; numpy float32 array of
        shape (n_particles,n_codims). Codims is almost always 3.
    forces:
        Forces of frames in system; numpy float32 array of
        shape (n_particles,n_codims). Codims is almost always 3.
    builder:
        flatbuffers.Builder instance or None; if None, a built in instance
        is used.
    check:
        Whether to run basic checks on argument shape and types.


    Returns:
    -------
    Bytes object that can be used to create an FBFrame instance.

    """
    # should validate argument type/shape

    if builder is None:
        builder = _builder

    if check:
        if len(forces.shape) != 2:
            raise ValueError(
                "Forces must be 2 dimensional (this object represents a single frame."
            )
        if len(positions.shape) != 2:
            raise ValueError(
                "Positions must be 2 dimensional (this object represents a single "
                "frame."
            )
        if positions.shape[0] != nsites:
            raise ValueError(
                "Positions have the wrong number of entries; "
                f"Expected {nsites}, found {positions.shape[0]}."
            )
        if forces.shape[0] != nsites:
            raise ValueError(
                "Forces have the wrong number of entries; "
                f"Expected {nsites}, found {forces.shape[0]}."
            )
        if positions.shape != forces.shape:
            raise ValueError("Forces and positions have differing shapes.")

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
