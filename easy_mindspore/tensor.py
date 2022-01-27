import numpy as np
from mindspore.common import dtype as mstype
from mindspore import Tensor

def tensor(data, *, dtype=None):
    if dtype is None:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.dtype == np.int64:
            data = data.astype(np.int32)
        if data.dtype == np.float64:
            data = data.astype(np.float32)
    return Tensor(data, dtype)
