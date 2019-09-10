from svtk.vtk_classes.vtk_animation_timer_callback import VTKAnimationTimerCallback
from pcoords.ento import EntoLine
from pcoords.neuralbinary import NeuralBinary
from svtk.vtk_classes.vtk_displayer import VTKDisplayer
import time

class EntoLineAnimator(VTKAnimationTimerCallback):
    def __init__(self):
        super(EntoLineAnimator, self).__init__()
        self.ento_line = EntoLine(2 ** 32 - 1)

    def loop(self, obj, event):
        super(EntoLineAnimator, self).loop(obj, event)


class ELACallbackClass(EntoLineAnimator):
    def __init__(self):
        super(ELACallbackClass, self).__init__()
        self.i = 0
        self.ento_callback = None

    def at_start(self):
        self.add_point_field(widths=[64, 25, 1],
                             normal=[0, 1, 0],
                             center=[0, 1, 0],
                             color=[[int(128), int(66), int(21)]])

    def set_ento_callback(self, callback):
        self.ento_callback = callback

    def loop(self, obj, event):
        super(ELACallbackClass, self).loop(obj, event)
        indexes = self.ento_line.get_indexes(self.i)
        if self.ento_callback:
            self.i = self.ento_callback(self.i)
        self.set_all_point_colors([int(0), int(0), int(0)])
        self.set_point_colors([int(255), int(255), int(255)],
                              [indexes[i] + i * self.ento_line.buckets for i in range(len(indexes))])


class NeuralBinaryAnimator(VTKAnimationTimerCallback):
    def __init__(self):
        super(NeuralBinaryAnimator, self).__init__()
        self.nb = NeuralBinary(0, min_val=.01, max_val=100)

    def loop(self, obj, event):
        super(NeuralBinaryAnimator, self).loop(obj, event)


class NBCallbackClass(NeuralBinaryAnimator):
    def __init__(self):
        super(NBCallbackClass, self).__init__()
        self.i = 0
        self.ento_callback = None

    def at_start(self):
        self.add_point_field(widths=[self.nb._array.shape[1], self.nb._array.shape[0], 1],
                             normal=[0, 1, 0],
                             center=[0, 1, 0],
                             color=[[int(128), int(66), int(21)]])

    def set_nb_callback(self, callback):
        self.nb_callback = callback

    def loop(self, obj, event):
        super(NBCallbackClass, self).loop(obj, event)
        if self.nb_callback:
            self.i = self.nb_callback(self.i)
        self.nb = NeuralBinary(self.i, min_val=.01, max_val=100)
        self.set_all_point_colors([int(0), int(0), int(0)])
        self.set_point_colors([int(255), int(255), int(255)],
                              list(self.nb.indexes()))


if __name__ == '__main__':
    ela = VTKDisplayer(NBCallbackClass)


    def ento_callback(i):
        return i+.01


    ela.callback_instance.set_nb_callback(ento_callback)
    ela.visualize()
