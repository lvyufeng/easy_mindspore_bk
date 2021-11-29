import numpy as np
from mindspore.common import dtype as mstype
from mindspore import Tensor as msTensor
from .ops import masked_fill_

class Tensor(msTensor):
    def __init__(self, input_data=None, dtype=mstype.float32, shape=None, init=None):
        super().__init__(input_data=input_data, dtype=dtype, shape=shape, init=init)

    def masked_fill(self, mask:"Tensor", value:float):
        return masked_fill_(self, mask, value)

def tensor(data, *, dtype=None):
    if dtype is None:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.dtype == np.int64:
            data = data.astype(np.int32)
        if data.dtype == np.float64:
            data = data.astype(np.float32)
    return Tensor(data, dtype)
