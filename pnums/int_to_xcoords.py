"""Translate int/float coordinates to formats more suitable for transformer neural networks."""

import math as m

import numpy as np

from pnums.ith_middle import ith_middle, mask_ith_middle
from typing import Dict, Any


def _int_to_packed_uint8(in_int):
    """
    Convert an int to a unit array.

    >>> _int_to_packed_uint8(16)
    array([16], dtype=uint8)

    """
    in_bytes = int(in_int).to_bytes(
        length=int(m.ceil(m.log2(in_int + 1) / 8.0)), byteorder="big", signed=False
    )
    in_array = np.frombuffer(in_bytes, dtype=np.uint8)
    return in_array


def int_to_bool_array(in_int, length):
    """
    Convert an int to a bool array.

    >>> int_to_bool_array(16, 32)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])

    """
    byte_arr = np.unpackbits(_int_to_packed_uint8(in_int))
    if len(byte_arr) < length:
        pad_l = np.zeros((length - len(byte_arr),))
        byte_arr = np.concatenate((pad_l, byte_arr))
    return byte_arr


def inversion_double(in_array):
    """
    Get the input boolean array along with its element-wise logical not beside it. For error correction.

    >>> inversion_double(np.array([1,0,1,1,1,0,0,1], dtype=np.bool))
    array([[ True, False,  True,  True,  True, False, False,  True],
           [False,  True, False, False, False,  True,  True, False]])
    """
    return np.stack((in_array, np.logical_not(in_array)))


_i_extend_dict: Dict[Any, Any] = {}  # optimization dictionary


def _i_extend(in_array, ext, sort=True):
    """
    Create `ext`*2 duplicates and shifts the duplicates so each number has a unique representation.

    Error correction could fetch back two numbers if they were recorded on the same array.
    Useful for neural algorithms that would guess multiple locations represented by these coordiantes.

    >>> _i_extend(np.array([ 1, 0, 1, 1, 1, 0, 0, 1], dtype=np.bool), 8).astype(np.uint8)
    array([[[1, 0, 1, 1, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 1, 0]],
    <BLANKLINE>
           [[1, 1, 1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 0]],
    <BLANKLINE>
           [[1, 1, 0, 1, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 1, 0]],
    <BLANKLINE>
           [[1, 1, 0, 1, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 1, 0]],
    <BLANKLINE>
           [[1, 1, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 1, 0, 1, 0, 0]],
    <BLANKLINE>
           [[0, 1, 0, 0, 1, 0, 1, 1],
            [1, 0, 1, 1, 0, 1, 0, 0]],
    <BLANKLINE>
           [[0, 1, 0, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 0, 0, 0]],
    <BLANKLINE>
           [[0, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 0, 0]]], dtype=uint8)

    """
    base_array = inversion_double(in_array)
    base_array = np.expand_dims(base_array, axis=0)

    if sort:
        if ext in _i_extend_dict.keys():
            i_list = _i_extend_dict[ext]
        else:
            i_list = [
                ith
                for _, ith in sorted(
                    zip(
                        [ith_middle(i) for i in range(ext - 1)],
                        [j for j in range(ext - 1)],
                    )
                )
            ]
            _i_extend_dict[ext] = i_list
    else:
        i_list = range(ext - 1)

    for i in i_list:
        next_array = mask_ith_middle(in_array, i)
        double_next_array = inversion_double(next_array)
        double_next_array = np.expand_dims(double_next_array, axis=0)
        base_array = np.concatenate((base_array, double_next_array), axis=0)
    return base_array


def int_to_1d(in_int, max_bits, new_dim_length):
    """Convert an integer to a 1D format with error correction for transformers."""
    bool_array = int_to_bool_array(in_int, max_bits)
    extended_bool_array = _i_extend(bool_array, new_dim_length)
    return extended_bool_array


def ints_to_2d(
    in_int_1, max_bits_1, new_dim_length_1, in_int_2, max_bits_2, new_dim_length_2
):
    """Convert two integers representing x,y coordinates to a correlated output for 2D transformers."""
    bool_ento_1 = int_to_1d(in_int_1, max_bits_1, new_dim_length_1).astype(np.bool)
    bool_ento_2 = int_to_1d(in_int_2, max_bits_2, new_dim_length_2).astype(np.bool)
    positive_array = np.zeros(
        (new_dim_length_1, new_dim_length_2, max(max_bits_1, max_bits_2)), dtype=np.bool
    )
    for i in range(bool_ento_1.shape[-1]):
        region_1 = bool_ento_1[:, 0, i, np.newaxis]
        region_2 = bool_ento_2[np.newaxis, :, 0, i]
        square_region = np.matmul(region_1, region_2)
        positive_array[:, :, i] = square_region
    return positive_array
