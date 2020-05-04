"""Translate coordinates made for transformer neural networks back into ints/floats."""

import numpy as np


def right_pack_array(in_array):
    """
    Packs a boolean array into uint8s. Unlike NumPy's packbits, this function pads to the left.

    >>> right_pack_array(np.array([0,0,1,1,0,1,0,0,1,1,0,1,0,0,1,0,1,0,1,0,1,0,1]))
    array([ 26, 105,  85], dtype=uint8)

    :param in_array:
    :return:
    """
    ext_in_array = np.zeros((8 - (len(in_array)) % 8,), dtype=np.bool)
    padded_in_array = np.concatenate((ext_in_array, in_array))
    packed_in = np.packbits(padded_in_array)
    return packed_in


def packed_unit8_to_int(in_array):
    """
    Packs a uint8 array into a single int.

    >>> packed_unit8_to_int(np.array([ 26, 105,  85], dtype=np.uint8))
    1730901

    :param in_array:
    :return:
    """
    num_i = int(0)
    for i, a in enumerate(reversed(in_array)):
        num_i += int(a) << (8 * i)
    return num_i


def right_pack_bool_to_int(in_array):
    """
    Convert an array of booleans to a packed array of unit8s.

    This is one part of getting numbers back out of pcoord format. Another is undoing the ith_middle change.

    >>> right_pack_bool_to_int(np.array([0,0,0,1,1,0,1,0,0,1,1,0,1,0,0,1,0,1,0,1,0,1,0,1]))
    1730901

    """
    arr1 = right_pack_array(in_array)
    arr2 = packed_unit8_to_int(arr1)
    return arr2
