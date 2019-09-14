from numbers import Real
import numpy as np
from pcoords.neuralfloat import NeuralFloat

class ProbabilisticNeuralFloat(Real):
    """

    """

    def __init__(self, min_val=2**-126, bit_duplicity=8):
        self.array = np.array([1], dtype=np.bool)

    def direct_combine(self, others, other_weights=[.5], self_weight=0.5):
        assert 0.0 < self_weight
        assert all([0.0 <= weight for weight in other_weights])
        total = sum(other_weights + [self_weight])
        self_weight = self_weight / total
        other_weights[:] = [weight / total for weight in other_weights]
        if isinstance(others[0], ProbabilisticNeuralFloat):
            if others.array.shape == self.array.shape:
                other_arrays = [other.array.astype(np.half) for other in others]
                self_array = self.array.astype(np.half)
                combined_array = self_array * self_weight
                for other_array, other_weight in zip(other_arrays, other_weights):
                    combined_array += other_array * other_weight
                combined_array = np.round(combined_array)
                combined_array = combined_array.astype(np.bool)
            else:
                raise IndexError("ProbabilisticNeuralFloat does not support different shaped arrays yet.")
        elif isinstance(others[0], NeuralFloat):
            if others.array.shape:
                pass

    def __float__(self):
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

    def __eq__(self, other):
        pass
