"""Probabilistic n-dimensional binary integer format."""

import itertools
from numbers import Real

import numpy as np
from scipy.ndimage import convolve  # type: ignore

from pnums.int_to_xcoords import int_to_bool_array, inversion_double
from pnums.xcoords_to_int import right_pack_bool_to_int


def shift(arr, num, fill_value=np.nan):
    """Shift data in array arr num positions left or right."""
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
    """Run an xor operation between two layers of neurons."""
    o = convolve(a, b, mode="wrap")
    s = np.sum(o)
    if s == 0:
        return layer_zero(o)
    else:
        return o / s


def layer_zero(a: np.ndarray):
    """Return the PInt representation of a 0 bit for a layer."""
    o = np.zeros_like(a)
    o[tuple(1 for _ in range(o.ndim))] = 1
    return o


def layer_one(a: np.ndarray):
    """Return the PInt representation of a 1 bit for a layer."""
    o = np.zeros_like(a)
    o[tuple(0 for _ in range(o.ndim))] = 1
    return o


def layer_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Run an and operation between two layers of neurons."""

    # algorithm should be an n-d product towards one. If one bit is 1,1,1...
    #  1,1,1 will multiply only with 1,1,1
    #  1,0,1 will multiply with 1,0,1 and 1,1,1
    #  1,0,0 will multiply with 100, 101, 110, 111
    # nd tensor algorithm will need a, b, and zero tensors
    one = layer_one(a)
    one = tuple(w[0] for w in np.where(one == 1))
    zero = layer_zero(a)
    zero = tuple(w[0] for w in np.where(zero == 1))
    o = np.zeros_like(a)

    # This first loop populates each location in a n-dimensional array where each dimension is length two, but it has a
    # flaw. If you're at 1,0 and the other is at 1,1, where 0,0 is the top left:
    #  0|1 and 0|0 = 0|1
    #  0|0     0|1   0|0
    # So you can't just multiply same positions. Instead, we sum up all different positions differing from true one in
    # all or less than the same dimension that our number is differing from one. Here are the diffs in 2d:
    # xy|x
    #  y|0
    # However, we discard zero, because it's never the result of AND with 1. So it's more like this in 2d:
    # `|x
    # y|0
    # And in 3D:
    # Top:  `|zx  Bottom: xy|x
    #      zy|z            y|0
    # So, this cube:
    # Top:  0|1  Bottom: 0|0
    #       0|0          0|0
    # Should be summed with all these positions:
    # Top:  0|1  Bottom: 0|1
    #       0|1          0|1
    # and then multiplied with the same position on b after b performs the same operation.
    for a_i, a_x in np.ndenumerate(a):
        a_d = []
        for e, i in enumerate(a_i):
            if i != one[e]:
                a_d.append(e)
        if len(a_d) > 0:
            a_sum = 0
            b_sum = 0
            for diff in itertools.product(*[[0, 1]] * len(a_d)):
                index = list(one)
                for d_e, d_d in zip(a_d, diff):
                    index[d_e] = d_d
                index = tuple(index)  # type: ignore
                a_sum += a[index]
                b_sum += b[index]

            o[a_i] += a_sum * b_sum
        else:
            o[a_i] += a[a_i] * b[one]

    # However, the problem with that last loop is that it add too much in the lower positions:
    #  0|0 and 0|0 = 0|1
    #  0|1     0|1   1|1
    # So now, we sum up each position closer to layer_one, and subtract that from any current position. Say this is the
    # current position:
    #  0|1
    #  0|0
    # then, this is the only one closer to one:
    #  0|0
    #  0|1
    # Which has a value of one. So you subtract 1 from the original 1 in that position, and get 0.
    for a_i, a_x in np.ndenumerate(a):
        a_d = []
        for e, i in enumerate(a_i):
            if i != one[e]:
                a_d.append(e)
        if len(a_d) > 0:
            for diff in itertools.product(*[[0, 1]] * len(a_d)):
                index = list(one)
                for d_e, d_d in zip(a_d, diff):
                    index[d_e] = d_d
                index = tuple(index)  # type: ignore
                if index != a_i:
                    o[a_i] -= o[index]

    # todo: these two loops can be defined, but the iteration has to be changed. Specifically, it has to start at
    #  layer_one, then go outward, iterating through all positions with one dimension difference, then two, etc.
    #  And for higher dimensions, this can be highly parallelizable. In 100 dimensions, there will be 100 items max
    #  with the same difference in dimensions from one.
    s = np.sum(o)
    if s == 0:
        return layer_zero(o)
    else:
        return o / s


def layer_not(a: np.ndarray) -> np.ndarray:
    """Run a not operation on a layer of neurons."""
    o = np.flip(a)
    return o


def layer_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Run an and operation between two layers of neurons."""

    # For now, I'll just cheat and use functional completeness.
    # A OR B = ( A NAND A ) NAND ( B NAND B )
    # layer_not is really quick, so this isn't expensive at all.

    a_not = layer_not(a)
    b_not = layer_not(b)
    an_and_bn = layer_and(a_not, b_not)
    an_nand_bn = layer_not(an_and_bn)

    # todo: make a faster version of this.
    return an_nand_bn


class PInt(Real):
    """Probabilistic representation of any number of correlated integers."""

    def __init__(self, *args, bits=32, confidence=1.0, dtype=np.float32):
        """
        Probabilistic representation of any number of correlated integers.

        :param args: a list of numbers or numpy array may be inputted
        :param bits: number of bits used to represent each number
        :param confidence: probability of this set of correlated numbers
        :param dtype: internal storage type
        """
        if isinstance(args[0], np.ndarray):
            self.tensor = args[0]
            self.ndim = self.tensor.ndim - 1
            self.bits = self.tensor.shape[-1]
        else:
            if dtype not in {
                np.float,
                np.float16,
                np.float32,
                np.float64,
                np.double,
                np.half,
            }:
                raise NotImplementedError(
                    "PInt currently only stores probabilities as values between 0 and 1. "
                    "Non-real types like int or complex will need different methods."
                )
            self.ndim = len(args)
            self.tensor = np.ones([2] * self.ndim + [bits], dtype=dtype)
            self.bits = bits

            for d in range(self.ndim):
                in_val = inversion_double(int_to_bool_array(args[d], bits)).astype(
                    np.float
                )
                if in_val.shape[-1] < bits:
                    last_slice = in_val.shape[-1]
                else:
                    last_slice = bits
                self.tensor[
                    (slice(2),) * d
                    + (0,)
                    + (slice(2),) * (self.ndim - d - 1)
                    + (slice(last_slice),)
                ] *= in_val[0, -bits:]
                self.tensor[
                    (slice(2),) * d
                    + (1,)
                    + (slice(2),) * (self.ndim - d - 1)
                    + (slice(last_slice),)
                ] *= in_val[1, -bits:]
                anti_tensor = (1 - self.tensor) * (
                    (1.0 - confidence) / (2 ** self.ndim - 1)
                )
            self.tensor *= confidence
            self.tensor += anti_tensor
        self.confidence = confidence

    def normalize(self, confidence=1.0):
        """Make each bit/layer add up to the same max probability."""
        new_tensor = np.zeros_like(self.tensor)
        for b in reversed(range(self.bits)):
            if np.sum(self.tensor[..., b]) != 0:
                new_tensor[..., b] = (
                    self.tensor[..., b] / np.sum(self.tensor[..., b]) * confidence
                )
        return PInt(new_tensor)

    def quantize(self):
        """Make each internal number one or zero. Useful for transforming back to normal numbers for computers."""
        new_tensor = np.zeros_like(self.tensor)
        for b in reversed(range(self.bits)):
            if np.sum(self.tensor[..., b]) != 0:
                max_indices = np.argwhere(
                    self.tensor[..., b] == np.amax(self.tensor[..., b])
                )
                ind = max_indices[0]
                if len(max_indices) == 1:
                    new_tensor[(*ind, b)] = 1
        return PInt(new_tensor)

    def overall_confidence(self):
        """Get the overall confidence of the entire probabilistic integer."""
        n = self.normalize()
        q = n.quantize().tensor
        n = n.tensor
        o = np.zeros_like(n)

        o[q > 0] = (n / q)[q > 0]

        sum = np.sum(o)
        avg = sum / np.count_nonzero(o)

        return avg

    def coalesce(self, value=1.0):
        """
        Interpolate between normalized and quantized PInts.

        Useful for keeping a stable representation, like Kalman filtering.
        """
        multiplier = self.quantize().tensor * value
        new_tensor = np.zeros_like(self.tensor)
        norm = self.normalize()
        for b in reversed(range(self.bits)):
            new_tensor[..., b] = multiplier[..., b] + (
                (1.0 - value) * norm.tensor[..., b]
            )
        return PInt(new_tensor)

    def _handle_other_types(self, other):
        if isinstance(other, PInt):
            other_tensor = other.tensor
        elif isinstance(other, np.ndarray):
            other_tensor = other
        else:
            raise NotImplementedError(f"{type(other)} type currently not implemented.")
        return other_tensor

    def __xor__(self, other):
        other_tensor = self._handle_other_types(other)
        self_confidence = self.confidence
        other_confidence = other.confidence
        half_sum_prob = np.zeros_like(self.tensor)
        for b in reversed(range(self.bits)):
            half_sum_prob[..., b] = layer_xor(self.tensor[..., b], other_tensor[..., b])
        return PInt(half_sum_prob).normalize(self_confidence + other_confidence)

    def __and__(self, other):
        other_tensor = self._handle_other_types(other)
        self_confidence = self.confidence
        other_confidence = other.confidence
        half_sum_prob = np.zeros_like(self.tensor)
        for b in reversed(range(self.bits)):
            half_sum_prob[..., b] = layer_and(self.tensor[..., b], other_tensor[..., b])
        return PInt(half_sum_prob).normalize(self_confidence + other_confidence)

    def __invert__(self):
        self_confidence = self.confidence
        half_sum_prob = np.zeros_like(self.tensor)
        for b in reversed(range(self.bits)):
            half_sum_prob[..., b] = layer_not(self.tensor[..., b])
        return PInt(half_sum_prob).normalize(self_confidence)

    def __or__(self, other):
        other_tensor = self._handle_other_types(other)
        self_confidence = self.confidence
        other_confidence = other.confidence
        half_sum_prob = np.zeros_like(self.tensor)
        for b in reversed(range(self.bits)):
            half_sum_prob[..., b] = layer_or(self.tensor[..., b], other_tensor[..., b])
        return PInt(half_sum_prob).normalize(self_confidence + other_confidence)

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
                carry_prob[..., b - 1] = layer_or(
                    layer_and(self.tensor[..., b], other_tensor[..., b]),
                    layer_and(half_sum_prob[..., b], carry_prob[..., b]),
                )

        return PInt(sum_prob).normalize(self_confidence + other_confidence)

    def __sub__(self, other):
        other_tensor = self._handle_other_types(other)
        return self.__add__(1 - other_tensor)

    def asfloat(self):
        """Return the floating point representation of all corellated numbers held in this PInt."""
        quant = self.normalize().quantize().tensor.astype(np.bool)
        if self.ndim == 1:
            val = right_pack_bool_to_int(quant[0, ...])
            return float(val)
        else:
            vals = []
            for n in range(self.ndim):
                pos_quant = quant[
                    tuple(slice(2) for _ in range(n))
                    + (0,)
                    + tuple(slice(2) for _ in range(n + 1, self.ndim))
                ]
                sum_quant = np.sum(
                    pos_quant, axis=tuple(i for i in range(self.ndim - 1))
                )
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
