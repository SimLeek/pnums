"""Ith middle functions for masking."""

import math as m
import numpy as np


def ith_middle(i, min_value=0.0, max_value=1.0):
    """
    Return the ith middle.

    >>> for i in range(10):
    ...   ith_middle(i)
    0.5
    0.25
    0.75
    0.125
    0.375
    0.625
    0.875
    0.0625
    0.1875
    0.3125

    """
    tree_level = m.floor(m.log2(i + 1))
    leaves_up_to_level = 2 ** tree_level
    id_in_level = i - leaves_up_to_level + 1
    value_for_level = (id_in_level + 0.5) / leaves_up_to_level
    middle_value = min_value + value_for_level * (max_value - min_value)
    return middle_value


def ith_binary_mask(i):
    """
    Return the ith binary mask for novel binary number representation.

    For one bit in a binary integer, take the bit and all bits below it until the length of the array is met.
    Add the resulting integer and this array together. The topmost bit will be one in the in a region around the
    ith middle. (Allow for overflow.)

    >>> for i in range(10):
    ...   ith_binary_mask(i)
    array([False,  True])
    array([False, False,  True])
    array([False,  True,  True])
    array([False, False, False,  True])
    array([False, False,  True,  True])
    array([False,  True, False,  True])
    array([False,  True,  True,  True])
    array([False, False, False, False,  True])
    array([False, False, False,  True,  True])
    array([False, False,  True, False,  True])

    """
    assert i < 256
    tree_level = m.floor(m.log2(i + 1))
    bit_mask = np.zeros((2 + tree_level), dtype=bool)
    bit_mask[-1] = 1
    leaves_up_to_level = 2 ** tree_level
    id_in_level = i - leaves_up_to_level + 1
    upper_bool = np.unpackbits(np.array([id_in_level], dtype=np.uint8))
    bit_mask[:-1] = upper_bool[-len(bit_mask) + 1 :]
    return bit_mask


def mask_ith_middle(in_array, i):
    """
    Transform a boolean array so its int form, a, is a*ith_middle(i) away.

    >>> mask_ith_middle(np.array([1,0,1,1,1,0,0,1], dtype=np.uint8), 0)
    array([1, 1, 0, 0, 1, 0, 1, 1], dtype=uint8)

    """
    i_array = ith_binary_mask(i)
    packed_i = np.packbits(i_array)
    ext_array = np.zeros((7,))
    padded_in_array = np.concatenate((in_array, ext_array)).astype(np.bool)
    return_array = np.zeros_like(in_array)
    for i in range(len(in_array)):
        pad_num = padded_in_array[i : i + 8]
        packed_num = np.packbits(pad_num)
        added_num = np.add(packed_num, packed_i)
        bool_added_num = np.unpackbits(added_num)
        return_array[i] = bool_added_num[0]
    return return_array


def unmask_ith_middle(in_array, i):
    """
    Transform a boolean array so its int form, a, is a*-ith_middle(i) away.

    #>>> unmask_ith_middle(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8), 0)
    array([1, 0, 1, 1, 1, 0, 0, 1], dtype=uint8)

    """
    raise NotImplementedError("Haven't figured this out yet.")
