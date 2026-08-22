import numpy as np
import pyarrow as pa


def to_fixed_list_float16(arr2d: np.ndarray, list_size: int) -> pa.Array:
    """Convert (B, list_size) float16 numpy array -> FixedSizeListArray<float16>."""
    flat = pa.array(arr2d.flatten(), type=pa.float16())
    return pa.FixedSizeListArray.from_arrays(flat, list_size)


def to_fixed_list_int8(arr2d: np.ndarray, list_size: int) -> pa.Array:
    """Convert (B, list_size) int8 numpy array -> FixedSizeListArray<int8>."""
    flat = pa.array(arr2d.flatten(), type=pa.int8())
    return pa.FixedSizeListArray.from_arrays(flat, list_size)
