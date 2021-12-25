from typing import Tuple, Union
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import constexpr

@constexpr
def _tuple_to_tensor(paddings):
    return Tensor(paddings)

class _ReflectionPadNd(nn.Cell):
    def construct(self, inputs, paddings):
        return ops.MirrorPad('REFLECT')(inputs, paddings)

class ReflectionPad1d(_ReflectionPadNd):
    def __init__(self, padding: Union[int, Tuple[int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

    def construct(self, inputs):
        ndim = inputs.ndim
        paddings = ((0, 0),) * (ndim - 1) + (self.padding,)
        paddings = _tuple_to_tensor(paddings)
        return super().construct(inputs, paddings)

class ReflectionPad2d(_ReflectionPadNd):
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        else:
            self.padding = padding

    def construct(self, inputs):
        ndim = inputs.ndim
        paddings = ((0, 0),) * (ndim - 2) + \
            ((self.padding[2], self.padding[3]),(self.padding[0], self.padding[1]),)
        paddings = _tuple_to_tensor(paddings)
        return super().construct(inputs, paddings)

class ReflectionPad3d(_ReflectionPadNd):
    def __init__(self, padding: Union[int, Tuple[int, int, int, int, int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding, padding, padding)
        else:
            self.padding = padding

    def construct(self, inputs):
        ndim = inputs.ndim
        paddings = ((0, 0),) + \
            ((self.padding[4], self.padding[5]), \
             (self.padding[2], self.padding[3]), \
             (self.padding[0], self.padding[1]),)
        paddings = _tuple_to_tensor(paddings)
        if ndim == 5:
            in_shape = inputs.shape
            reshaped_inputs = inputs.reshape(-1, *in_shape[2:])
            outputs = super().construct(reshaped_inputs, paddings)
            return outputs.reshape(*in_shape[:2], *outputs.shape[1:])
        return super().construct(inputs, paddings)

class ZeroPad2d(nn.Cell):
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        super().__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        else:
            self.padding = padding

    def construct(self, inputs):
        ndim = inputs.ndim
        paddings = ((0, 0),) * (ndim - 2) + \
            ((self.padding[2], self.padding[3]),(self.padding[0], self.padding[1]),)
        return ops.Pad(paddings)(inputs)