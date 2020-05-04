from pnums.int_to_xcoords import int_to_bool_array, inversion_double
import numpy as np
import itertools

def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[..., :num] = fill_value
        result[..., num:] = arr[..., :-num]
    elif num < 0:
        result[..., num:] = fill_value
        result[..., :num] = arr[..., -num:]
    else:
        result[:] = arr
    return result

class PInt(object):
    def __init__(self, *args, bits=32, confidence=1.0, dtype=np.float32):
        if isinstance(args[0], np.ndarray):
            self.tensor = args[0]
            self.ndim = self.tensor.ndim - 1
            self.bits = self.tensor.shape[-1]
        else:
            if dtype not in {np.float, np.float16, np.float32, np.float64, np.double, np.half}:
                raise NotImplementedError("PInt currently only stores probabilities as values between 0 and 1. "
                                          "Non-real types like int or complex will need different methods.")
            self.ndim = len(args)
            self.tensor = np.ones([2] * self.ndim + [bits], dtype=dtype)
            self.bits = bits

            for d in range(self.ndim):
                in_val = inversion_double(int_to_bool_array(args[d], bits)).astype(np.float)
                if in_val.shape[-1]<bits:
                    last_slice = in_val.shape[-1]
                else:
                    last_slice = bits
                self.tensor[[slice(2)] * d + [0] + [slice(2)] * (self.ndim - d - 1) + [slice(last_slice)]] *= in_val[0, -bits:]
                self.tensor[[slice(2)] * d + [1] + [slice(2)] * (self.ndim - d - 1) + [slice(last_slice)]] *= in_val[1, -bits:]
            anti_tensor = (1 - self.tensor) * ((1.0 - confidence)/(2**self.ndim-1))
            self.tensor *= confidence
            self.tensor += anti_tensor

    def __add__(self, other):
        """Full adder implemented with probabilities."""
        if isinstance(other, PInt):
            other_tensor = other.tensor
        elif isinstance(other, np.ndarray):
            other_tensor = other
        else:
            raise NotImplementedError("")

        half_sum_prob = np.zeros_like(self.tensor)
        sum_prob = np.zeros_like(self.tensor)
        carry_prob = np.zeros_like(self.tensor)
        for b in reversed(range(self.bits)):
            for pos in itertools.product(*([[0, 1]] * self.ndim)):
                # todo: replace these 'xor's with convolutions
                half_sum_prob[(*pos, b)] += (self.tensor[(*pos, b)] * (1 - other_tensor[(*pos, b)])) + ((1 - self.tensor[(*pos, b)]) * other_tensor[(*pos, b)])  # probabilistic xor
                sum_prob[(*pos, b)] += (half_sum_prob[(*pos, b)] * (1 - carry_prob[(*pos, b)])) + ((1 - half_sum_prob[(*pos, b)]) * carry_prob[(*pos, b)])  # probabilistic xor
                if b > 0:
                    carry_prob[(*pos, b - 1)] += self.tensor[(*pos, b)] * other_tensor[(*pos, b)] + half_sum_prob[(*pos, b)] * carry_prob[(*pos, b)]
        for b in reversed(range(self.bits)):
            if np.sum(sum_prob[(..., b)]) != 0:
                sum_prob[(..., b)] /= np.sum(sum_prob[(..., b)])

        return PInt(sum_prob)

    def __sub__(self, other):
        if isinstance(other, PInt):
            other_tensor = other.tensor
        elif isinstance(other, np.ndarray):
            other_tensor = other
        else:
            raise NotImplementedError("")

        return self.__add__(1 - other_tensor)
