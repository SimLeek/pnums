"""Work In Progress."""

from svtk.vtk_classes.vtk_animation_timer_callback import VTKAnimationTimerCallback

import pnums.int_to_xcoords
from pnums.neuralfloat import NeuralFloat
from svtk.vtk_classes.vtk_displayer import VTKDisplayer
import time

yellow_dark = [152, 122, 38]
yellow_bright = [255, 238, 188]

red_dark = [152, 57, 38]
red_bright = [255, 200, 188]

brown_dark = [152, 95, 37]
brown_bright = [255, 222, 188]


class NeuralBinaryAnimator(VTKAnimationTimerCallback):
    def __init__(self):
        super(NeuralBinaryAnimator, self).__init__()
        self.nb = NeuralFloat(0, min_val=0.01, max_val=2 ** 100)

    def loop(self, obj, event):
        super(NeuralBinaryAnimator, self).loop(obj, event)


class NBCallbackClass(NeuralBinaryAnimator):
    def __init__(self):
        super(NBCallbackClass, self).__init__()
        self.i = 1.1
        self.ento_callback = None

    def at_start(self):
        self.add_point_field(
            widths=[self.nb._array.shape[1], self.nb._array.shape[0], 1],
            normal=[0, 1, 0],
            center=[0, 1, 0],
            color=[brown_dark],
        )

    def set_nb_callback(self, callback):
        self.nb_callback = callback

    def loop(self, obj, event):
        super(NBCallbackClass, self).loop(obj, event)
        if self.nb_callback:
            self.i = self.nb_callback(self.i)
        self.nb = NeuralFloat(self.i, min_val=0.01, max_val=2 ** 100)
        self.set_all_point_colors(brown_dark)
        self.set_point_colors(brown_bright, list(self.nb.indexes()))


import numpy as np


class RobustBinaryAnimator(VTKAnimationTimerCallback):
    def __init__(self):
        super(RobustBinaryAnimator, self).__init__()
        self.nb = pnums.int_to_xcoords.ints_to_2d(0, 32, 8, 0, 32, 8)

    def loop(self, obj, event):
        super(RobustBinaryAnimator, self).loop(obj, event)


class RobustCallbackClass(RobustBinaryAnimator):
    def __init__(self):
        super(RobustCallbackClass, self).__init__()
        self.i = 1.1
        self.robust_callback = None

    def at_start(self):
        """Setup the point field at the start."""
        self.add_point_field(
            widths=[self.nb.shape[2], self.nb.shape[1], self.nb.shape[0]],
            normal=[0, 1, 0],
            center=[0, 1, 0],
            color=[brown_dark],
        )

    def set_nb_callback(self, callback):
        """Set the callback."""
        self.robust_callback = callback

    def loop(self, obj, event):
        """Run the main loop."""
        super(RobustCallbackClass, self).loop(obj, event)
        if self.robust_callback:
            self.i = self.robust_callback(self.i)
        self.nb = pnums.int_to_xcoords.ints_to_2d(
            self.i, 32, 8, self.i / 3, 32, 8
        )
        self.set_all_point_colors(brown_dark)
        indexes = np.nonzero(np.ravel(self.nb))
        self.set_point_colors(brown_bright, list(indexes))


if __name__ == "__main__":
    ela = VTKDisplayer(RobustCallbackClass)
    ela.point_size = 10

    ento_callback = lambda i: int((i + 1) % 2 ** 32)

    ela.callback_instance.set_nb_callback(ento_callback)
    ela.visualize()
