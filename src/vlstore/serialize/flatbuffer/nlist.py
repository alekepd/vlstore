"""Scratch script for trying flatbuffers."""
from typing import Final, Any, Optional, Union, overload, Literal
from dataclasses import dataclass
import flatbuffers  # type: ignore
from flatbuffers import number_types as N
import numpy as np
from .FBSites import FBNList
from ...store.util import bytewise_memoryview
from .._types import Codec, T_OUT, TYPE_OUT, TYPE_RETURNDATA, TYPE_INPUTDATA

# Specifies default starting size for builder
INI_SIZE: Final = int(2**20)
# buffer object that is used to create flatbuffer objects if none is given
_builder = flatbuffers.Builder(INI_SIZE)


@dataclass(frozen=True)
class BackedNList:
    """Neighbor list of single frame.

    This object is backed by an FBNList instance; most attributes are properties
    which access and transform the underlying data. It is also the most convenient
    way to create an FBNlist instance.

    Note that __init__ takes an already formed FBNList instance; for easy creation,
    see alternative init .from_values.

    Attributes:
    ----------
    raw:
        Buffer containing memory used by underlying FBFrame.
    fb:
        Underlying FBFrame instance.

    """

    raw: memoryview
    fb: FBNList.FBNList

    @property
    def tag(self) -> str:
        """Tag."""
        return self.fb.Tag().decode("utf-8")

    @property
    def order(self) -> int:
        """Order."""
        return self.fb.Order()

    @property
    def rcut(self) -> float:
        """Cutoff radius."""
        return self.fb.Rcut()

    @property
    def self_interaction(self) -> bool:
        """Flag specifying whether self interactions are included."""
        return self.fb.SelfInteraction()

    @property
    def index_mapping(self) -> np.ndarray:
        """Array specifying neighbor indices."""
        data = self.fb.IndexMappingAsNumpy()
        order = self.fb.Order()
        return data.reshape((order, -1))

    @property
    def cell_shifts(self) -> np.ndarray:
        """Array specifying cell shifts."""
        data = self.fb.CellShiftsAsNumpy()
        return data.reshape((-1, 3))

    @classmethod
    def from_values(cls, **kwargs) -> "BackedNList":  # noqa
        """Create instance by passing defining values.

        Values are transformed into an underlying flatbuffer object and used
        to create an instance.

        (The following are likely arguments, see create_fbnlist_buffer for details)

        Arguments:
        ---------
        name: str
            Specifies name of frame (e.g., the name of the protein)
        builder: Optional[flatbuffers.Builder] = None

        Returns:
        -------
        Instance of BackedAtomicData

        """
        raw = bytewise_memoryview(create_fbnlist_buffer(**kwargs))
        fb = FBNList.FBNList.GetRootAs(raw, 0)
        return cls(raw, fb)


@overload
def _pack_BackedNList(
    target: BackedNList,
    /,
    *,
    out: Literal[None] = ...,
) -> TYPE_INPUTDATA:
    ...


@overload
def _pack_BackedNList(
    target: BackedNList,
    /,
    *,
    out: T_OUT,
) -> T_OUT:
    ...


def _pack_BackedNList(
    target: BackedNList,
    out: Optional[TYPE_OUT] = None,
) -> Union[TYPE_INPUTDATA, TYPE_OUT]:
    """Serialize BackedNList instance.

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


def _unpack_BackedNList(data: TYPE_RETURNDATA) -> BackedNList:
    """Create BackedNList from buffer.

    Creates an FBBackedNList instance and then a BackedNList instance.
    """
    view = bytewise_memoryview(data)
    return BackedNList(view, FBNList.FBNList.GetRootAs(view, 0))


# This is used to pack and unpack BackedNList instances.
BackedNListCodec = Codec(pack=_pack_BackedNList, unpack=_unpack_BackedNList)


def create_fbnlist_buffer(
    tag: str,
    order: int,
    rcut: float,
    self_interaction: bool,
    index_mapping: np.ndarray,
    cell_shifts: np.ndarray,
    builder: Optional[flatbuffers.Builder] = None,
    check: bool = True,
) -> bytes:
    """Create a bytes object containing FBNList.

    Note that some of the argument names disagree in formatting relative to
    BackedNList; Consider using BackedNlist.from_values instead.

    This method is slow because the FlatBuffer python interface is slow.

    Arguments:
    ---------
    tag:
        String identifying neighbor list calculation.
    order:
        Order of neighborlist calculation.
    rcut:
        Cut off radius of neighborlist calculation.
    self_interaction:
        Whether self interactions are included in neighborlist.
    index_mapping:
        Array of neighboring indices; should be a of shape (2,any).
    cell_shifts:
        Array of cell shifts; should be a of shape (any, 3).
    builder:
        flatbuffers.Builder instance or None; if None, a built in instance
        is used.
    check:
        Whether to run basic checks on argument shape and types.


    Returns:
    -------
    Bytes object that can be used to create an FBNList instance.

    """
    # should validate argument type/shape

    if builder is None:
        builder = _builder

    if check:
        if len(index_mapping.shape) != 2:
            raise ValueError("index_mapping must be 2 dimensional.")
        if len(cell_shifts.shape) != 2:
            raise ValueError("cell_shifts must be two dimensional.")
        if index_mapping.shape[0] != order:
            raise ValueError(
                "index_mapping has the wrong shape in the first entry; "
                f"Expected {order}, found {index_mapping.shape[0]}."
            )
        if cell_shifts.shape[1] != 3:
            raise ValueError(
                "cell_shifts has the wrong shape in the second entry; "
                f"Expected 3, found {cell_shifts.shape[1]}."
            )

    index_mapping_offset = add_index_mapping_vector(
        builder, data=index_mapping.flatten().tolist(), fbclass=FBNList
    )

    cell_shifts_offset = add_cell_shifts_vector(
        builder, data=cell_shifts.flatten().tolist(), fbclass=FBNList
    )

    tag_string = builder.CreateString(tag)

    FBNList.Start(builder)

    FBNList.AddTag(builder, tag_string)
    FBNList.AddOrder(builder, order)
    FBNList.AddRcut(builder, rcut)
    FBNList.AddSelfInteraction(builder, self_interaction)
    FBNList.AddIndexMapping(builder, index_mapping_offset)
    FBNList.AddCellShifts(builder, cell_shifts_offset)

    frame_offset = FBNList.End(builder)

    builder.Finish(frame_offset)

    content = builder.Output()

    builder.Clear()
    return content


def add_index_mapping_vector(
    builder: flatbuffers.Builder, data: np.ndarray, fbclass: Any
) -> int:
    """Add index_mapping (int64) vector."""
    fbclass.StartPositionsVector(builder, len(data))
    reversed_data = reversed(data)
    # typically, the call should use: _call = builder.PrependInt64
    # we can get minor speed increases by inlining the corresponding operations,
    # but this comes at a risk.
    _call = builder.Prepend

    # can probably shave off some lookups
    # https://github.com/google/flatbuffers/blob/dafd2f1f29c5c0c2a6df285e9efe5c49746c6b7a
    # /python/flatbuffers/builder.py#L638

    dtype = N.Int64Flags
    for i in reversed_data:
        _call(dtype, i)
    return builder.EndVector()


def add_cell_shifts_vector(
    builder: flatbuffers.Builder, data: np.ndarray, fbclass: Any
) -> int:
    """Add cell_shifts (float32) vector."""
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
