"""Work in progress."""
import math
import warnings
import numpy as np
from numbers import Real


def _binlist_to_int(binlist):
    """
    Convert a binary list to int.

    Do not use. This should be replaces with the similar function in xcoords_to_int.
    """
    # from: https://stackoverflow.com/a/12461400
    out = 0
    for bit in binlist:
        out = (out << 1) | int(bit)
    return out


class NeuralFloat(Real):
    """
    Converts A floating point number to a binary array that's closer to what the brain uses, centered at zero.

    The benefit of using this is that similar numbers have a lot of similar bits on (or "activated neurons"), so
    networks that connect to this number to learn will be able to learn be able to learn ranges. Or areas for the 2D
    version of this.

    >>> a = NeuralFloat(3.14, min_val=.001, max_val=100)
    >>> print(a)
    [[0. 0. 0. 0. 0. 1. 1. 0. 0. 1. 0. 0. 0. 1. 1. 1. 1. 0. 1. 0. 1. 1. 1. 0.
      0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 1. 1. 1. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0. 1.
      1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    >>> print(float(a))
    3.1399998664855957

    >>> b = NeuralFloat(3.14*3, min_val=.001, max_val=100)
    >>> print(b)
    [[0. 0. 0. 1. 0. 0. 1. 0. 1. 1. 0. 1. 0. 1. 1. 1. 0. 0. 0. 0. 1. 0. 1. 0.
      0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 1. 1. 0. 1. 0. 0. 1. 0. 1. 0. 0. 0. 1. 1. 1. 1. 0. 1. 0. 1.
      1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    >>> print(float(b))
    9.419998168945312
    """

    def __init__(self, num=0.0, min_val=2 ** -126, max_val=2 ** 127, mantissa_bits=23):
        """Create a 32-bit neural float based on the input number."""
        self.float_val = num
        self.down_bits = int(math.ceil(math.log2(1 / min_val)))
        if self.down_bits >= 127:
            warnings.warn(
                "Allowing minimum possible float values will require many bits. Please limit if possible."
            )
        self.up_bits = int(math.ceil(math.log2(max_val)))
        if self.up_bits >= 126:
            warnings.warn(
                "Allowing maximum possible float valuse will require many bits. Please limit if possible."
            )
        self.mantissa_bits = int(mantissa_bits)
        self.bits = self.mantissa_bits + self.up_bits + self.down_bits

        self._array = self.convert_float_to_nb32(num)

    def convert_float_to_nb32(self, float_in):
        """Convert an ieee float to 32-bit neural binary float."""
        z = np.zeros((2, self.bits))
        mantissa, exp = math.frexp(float_in)
        if mantissa < 0:
            sign = 0
        else:
            sign = 1
        if exp > self.up_bits:
            exp = self.up_bits
            mantissa_array = [1 for _ in range(self.mantissa_bits)]
        elif exp < -self.down_bits:
            exp = -self.down_bits
            mantissa_array = [0 for _ in range(self.mantissa_bits)]
        else:
            mantissa_array = [
                int(x) for x in bin(int(abs(mantissa * 2 ** self.mantissa_bits)))[2:]
            ]
        active_bits = np.array(mantissa_array, dtype=np.bool)
        z[0, self.up_bits - exp : self.up_bits - exp + self.mantissa_bits] = active_bits
        z[1, self.up_bits - exp : self.up_bits - exp + self.mantissa_bits] = np.invert(
            active_bits
        )
        z[0, -1] = sign
        z[1, -1] = not sign
        return z

    def __float__(self):
        start = min(np.nonzero(self._array == 1)[1])
        mantissa_array = self._array[0, start : start + self.mantissa_bits]
        mantissa = _binlist_to_int(mantissa_array) / (2 ** self.mantissa_bits)
        exp = self.up_bits - start
        msign = self._array[1, -1]
        psign = self._array[0, -1]
        out = (msign * -1 + psign * 1) * mantissa * 2.0 ** exp
        return out

    def indexes(self):
        """Get the nonzero indexes of the float."""
        return np.nonzero(np.ravel(self._array))

    def __str__(self):
        return str(self._array)

    def __repr__(self):
        return str(self)

    def __add__(self, other):
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

    def __trunc__(self):
        pass

    def __floor__(self):
        pass

    def __ceil__(self):
        pass

    def __round__(self, ndigits=None):
        pass

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

    def __eq__(self, other):
        pass


class NeuralRobustFloat(Real):
    """
    Uses a list of masks to convert a NeuralFloat to a set of new representations, which can all be unmasked.

    The benefit of using this is that if the binary values are combined later, multiple numbers can be taken out
    instead of one

    >>> a = NeuralFloat(3.14, min_val=.001, max_val=100)
    >>> print(a)
    [[0. 0. 0. 0. 0. 1. 1. 0. 0. 1. 0. 0. 0. 1. 1. 1. 1. 0. 1. 0. 1. 1. 1. 0.
      0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 1. 1. 1. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0. 1.
      1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    >>> print(float(a))
    3.1399998664855957

    >>> b = NeuralFloat(3.14*3, min_val=.001, max_val=100)
    >>> print(b)
    [[0. 0. 0. 1. 0. 0. 1. 0. 1. 1. 0. 1. 0. 1. 1. 1. 0. 0. 0. 0. 1. 0. 1. 0.
      0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 1. 1. 0. 1. 0. 0. 1. 0. 1. 0. 0. 0. 1. 1. 1. 1. 0. 1. 0. 1.
      1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    >>> print(float(b))
    9.419998168945312
    """

    pass
