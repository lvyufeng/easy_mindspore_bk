import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp

class Reverse(nn.Cell):
    """Reverse operator, like Reverse in mindspore"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def construct(self, input_x):
        shape = input_x.shape
        dim_size = shape[self.dim]
        reversed_indexes = mnp.arange(dim_size-1, -1, -1)
        output = ops.Gather()(input_x, reversed_indexes, self.dim)
        return output