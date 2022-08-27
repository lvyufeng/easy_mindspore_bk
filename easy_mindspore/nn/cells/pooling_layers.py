import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import constexpr

@constexpr
def compute_kernel_size(inp_shape, output_size):
    kernel_width, kernel_height = inp_shape[2], inp_shape[3]
    if isinstance(output_size, int):
        kernel_width = math.ceil(kernel_width / output_size) 
        kernel_height = math.ceil(kernel_height / output_size)
    elif isinstance(output_size, list) or isinstance(output_size, tuple):
        kernel_width = math.ceil(kernel_width / output_size[0]) 
        kernel_height = math.ceil(kernel_height / output_size[1])
    return (kernel_width, kernel_height)


class MaxPool1d(nn.Cell):
    def __init__(self, auto_prefix=True, flags=None):
        super().__init__(auto_prefix=auto_prefix, flags=flags)

class MaxPool2d(nn.Cell):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.max_pool = ops.MaxPool(kernel_size, stride)
        self.use_pad = padding != 0
        if isinstance(padding, tuple):
            assert len(padding) == 2
            paddings = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        elif isinstance(padding, int):
            paddings = ((0, 0),) * 2 + ((padding, padding),) * 2
        else:
            raise ValueError('padding should be a tuple include 2 numbers or a int number')
        self.pad = ops.Pad(paddings)
    
    def construct(self, x):
        if self.use_pad:
            x = self.pad(x)
        return self.max_pool(x)

class MaxPool3d(nn.Cell):
    def __init__(self, auto_prefix=True, flags=None):
        super().__init__(auto_prefix=auto_prefix, flags=flags)

class AvgPool1d(nn.Cell):
    def __init__(self, auto_prefix=True, flags=None):
        super().__init__(auto_prefix=auto_prefix, flags=flags)

class AvgPool2d(nn.Cell):
    def __init__(self, auto_prefix=True, flags=None):
        super().__init__(auto_prefix=auto_prefix, flags=flags)

class AvgPool3d(nn.Cell):
    def __init__(self, auto_prefix=True, flags=None):
        super().__init__(auto_prefix=auto_prefix, flags=flags)

class AdaptiveMaxPool1d(nn.Cell):
    def __init__(self, output_size):
        super().__init__()

class AdaptiveMaxPool2d(nn.Cell):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
    
    def construct(self, x):
        inp_shape = x.shape
        kernel_size = compute_kernel_size(inp_shape, self.output_size)
        return ops.MaxPool(kernel_size, kernel_size)(x)

class AdaptiveMaxPool3d(nn.Cell):
    def __init__(self, output_size):
        super().__init__()

class AdaptiveAvgPool1d(nn.Cell):
    def __init__(self, output_size):
        super().__init__()

class AdaptiveAvgPool2d(nn.Cell):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
    
    def construct(self, x):
        inp_shape = x.shape
        kernel_size = compute_kernel_size(inp_shape, self.output_size)
        return ops.AvgPool(kernel_size, kernel_size)(x)

class AdaptiveAvgPool3d(nn.Cell):
    def __init__(self, output_size):
        super().__init__()