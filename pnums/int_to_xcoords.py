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
