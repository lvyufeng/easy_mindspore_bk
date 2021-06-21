import mindspore
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.numpy as mnp
from mindspore import Tensor

class MaskedFill(nn.Cell):
    def __init__(self, value):
        super(MaskedFill, self).__init__()
        self.value = value
        self.select = P.Select()
        self.cast = P.Cast()

    def construct(self, inputs:Tensor, mask:Tensor):
        if mask.dtype != mindspore.bool_:
            mask = self.cast(mask, mindspore.bool_)
        masked = mnp.full_like(inputs, self.value)
        outputs = self.select(mask, masked, inputs)
        return outputs