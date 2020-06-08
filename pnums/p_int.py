from numbers import Real

import numpy as np
from scipy.ndimage import convolve

from pnums.int_to_xcoords import int_to_bool_array, inversion_double
from pnums.xcoords_to_int import right_pack_bool_to_int


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


def layer_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    o = convolve(a, b, mode='wrap')
    s = np.sum(o)
    if s != 0:
        return o / s
    else:
        return layer_zero(o)


def layer_zero(a: np.ndarray):
    o = np.zeros_like(a)
    o[tuple(1 for _ in range(o.ndim))] = 1
    return o


def layer_one(a: np.ndarray):
    o = np.zeros_like(a)
    o[tuple(0 for _ in range(o.ndim))] = 1
    return o


def layer_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim > 1:
        a_max = np.argwhere(np.max(a) == a)
        b_max = np.argwhere(np.max(b) == b)
        c_sum = np.sum((1 - a_max) * (1 - b_max))
        a_sum = np.sum(a_max)
        b_sum = np.sum(b_max)
        if a_sum > b_sum:
            o = a * np.sum(b)
        elif a_sum == b_sum or c_sum == 0:
            o = a * b
        else:
            o = b * np.sum(a)
    else:
        o = a * b
    s = np.sum(o)
    if s != 0:
        return o / s
    else:
        return layer_zero(o)


def layer_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    o = convolve(a, b, mode='constant', cval=0)
    s = np.sum(o)
    if s != 0:
        return o / s
    else:
        return layer_zero(o)


class PInt(Real):
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
                if in_val.shape[-1] < bits:
                    last_slice = in_val.shape[-1]
                else:
                    last_slice = bits
                self.tensor[
                    (slice(2),) * d + (0,) + (slice(2),) * (self.ndim - d - 1) + (slice(last_slice),)] *= in_val[0,
                                                                                                          -bits:]
                self.tensor[
                    (slice(2),) * d + (1,) + (slice(2),) * (self.ndim - d - 1) + (slice(last_slice),)] *= in_val[1,
                                                                                                          -bits:]
                anti_tensor = (1 - self.tensor) * ((1.0 - confidence) / (2 ** self.ndim - 1))
            self.tensor *= confidence
            self.tensor += anti_tensor
        self.confidence = confidence

    def normalize(self, confidence=1.0):
        new_tensor = np.zeros_like(self.tensor)
        for b in reversed(range(self.bits)):
            if np.sum(self.tensor[..., b]) != 0:
                new_tensor[..., b] = self.tensor[..., b] / np.sum(self.tensor[..., b]) * confidence
        return PInt(new_tensor)

    def quantize(self):
        new_tensor = np.zeros_like(self.tensor)
        for b in reversed(range(self.bits)):
            if np.sum(self.tensor[..., b]) != 0:
                max_indices = np.argwhere(self.tensor[..., b] == np.amax(self.tensor[..., b]))
                ind = max_indices[0]
                if len(max_indices) == 1:
                    new_tensor[(*ind, b)] = 1
        return PInt(new_tensor)

    def coalesce(self, value=1.0):
        multiplier = self.quantize().tensor * value
        new_tensor = np.zeros_like(self.tensor)
        norm = self.normalize()
        for b in reversed(range(self.bits)):
            new_tensor[..., b] = multiplier[..., b] + ((1.0 - value) * norm.tensor[..., b])
        return PInt(new_tensor)

    def _handle_other_types(self, other):
        if isinstance(other, PInt):
            other_tensor = other.tensor
        elif isinstance(other, np.ndarray):
            other_tensor = other
        else:
            raise NotImplementedError(f"{type(other)} type currently not implemented.")
        return other_tensor

    def __add__(self, other):
        """Full adder implemented with probabilities."""
        other_tensor = self._handle_other_types(other)
        self_confidence = self.confidence
        other_confidence = other.confidence
        half_sum_prob = np.zeros_like(self.tensor)
        sum_prob = np.zeros_like(self.tensor)
        carry_prob = np.zeros_like(self.tensor)
        carry_prob[tuple(1 for _ in range(self.ndim)) + (self.bits - 1,)] = 1
        for b in reversed(range(self.bits)):
            half_sum_prob[..., b] = layer_xor(self.tensor[..., b], other_tensor[..., b])
            sum_prob[..., b] = layer_xor(half_sum_prob[..., b], carry_prob[..., b])
            if b > 0:
                carry_prob[..., b - 1] = layer_or(layer_and(self.tensor[..., b], other_tensor[..., b]),
                                                  layer_and(half_sum_prob[..., b], carry_prob[..., b]))

        return PInt(sum_prob).normalize(self_confidence + other_confidence)

    def __sub__(self, other):
        other_tensor = self._handle_other_types(other)
        return self.__add__(1 - other_tensor)

    def asfloat(self):
        quant = self.normalize().quantize().tensor.astype(np.bool)
        if self.ndim == 1:
            val = right_pack_bool_to_int(quant[0, ...])
            return float(val)
        else:
            vals = []
            for n in range(self.ndim):
                pos_quant = quant[
                    tuple(slice(2) for _ in range(n)) + (0,) + tuple(slice(2) for _ in range(n + 1, self.ndim))]
                sum_quant = np.sum(pos_quant, axis=tuple(i for i in range(self.ndim - 1)))
                val = right_pack_bool_to_int(sum_quant)
                vals.append(val)
            return tuple(vals)

    def __float__(self):
        f = self.asfloat()
        if isinstance(f, float):
            return f
        elif isinstance(f, tuple):
            val = 0
            for n, i in enumerate(f):
                val += n * (2 ** self.bits) ** i
            return val

    def __trunc__(self):
        f = self.asfloat()
        if isinstance(f, float):
            return int(f)
        elif isinstance(f, tuple):
            val = 0
            for n, i in enumerate(f):
                val += n * (2 ** self.bits) ** i
            return int(val)

    def __floor__(self):
        return self.__trunc__()

    def __ceil__(self):
        return self.__trunc__()

    def __round__(self, ndigits=None):
        return self.__trunc__()

    def __floordiv__(self, other):
        pass

    def __rfloordiv__(self, other):
        pass

    def __mod__(self, other):
        pass

    def __rmod__(self, other):
        pass

    def __lt__(self, other):
        pass

    def __le__(self, other):
        pass

    def __radd__(self, other):
        pass

    def __neg__(self):
        pass

    def __pos__(self):
        pass

    def __mul__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    def __pow__(self, exponent):
        pass

    def __rpow__(self, base):
        pass

    def __abs__(self):
        pass

    def __eq__(self, other):
        pass
