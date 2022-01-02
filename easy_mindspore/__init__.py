import mindspore
import mindspore.numpy as mnp
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.ops import constexpr
from mindspore._checkparam import Validator as validator
from typing import overload, Union, List, Tuple
# from . import callbacks, core, datasets, initializers, layers, metrics, operators
from .tensor import *
from .core.api import *

_size = Union[List[int], Tuple[int, ...]]

def abs(x, dtype=None):
    return mnp.abs(x, dtype)

def norm(x, ord=None, axis=None, keepdims=False):
    return mnp.norm(x, ord, axis, keepdims)

def matmul(x1, x2, dtype=None):
    return mnp.matmul(x1, x2, dtype)

def sum(data, axis=None, dtype=None):
    return mnp.sum(data, axis, dtype)

def dot(x, y):
    return mnp.dot(x, y)

def exp(data):
    return mnp.exp(data, mstype.float32)

def size(data):
    return mnp.size(data)

def zeros(shape, dtype=mstype.float32):
    return mnp.zeros(shape, dtype)

def zeros_like(data, dtype=None):
    return mnp.zeros_like(data, dtype)

def ones(shape, dtype=mstype.float32):
    return mnp.ones(shape, dtype)

def ones_like(data, dtype=None, shape=None):
    return mnp.ones_like(data, dtype=None)

def arange(start, stop=None, step=None, dtype=None):
    return mnp.arange(start, stop, step, dtype)

def linspace(start, stop, num, dtype=None):
    return mnp.linspace(start, stop, num, dtype=dtype)

def logspace(start, stop, num, base=10.0, dtype=mstype.float32):
    return mnp.logspace(start, stop, num, base=base, dtype=dtype)

def eye(n, m=None, dtype=mstype.float32):
    return mnp.eye(n, m, dtype=dtype)

def concat(data, axis=0):
    return mnp.concatenate(data, axis=axis)

def split(x, indices_or_sections, axis=0):
    return mnp.split(x, indices_or_sections, axis)

def masked_select():
    pass

def narrow():
    pass

def transpose(a, axes=None):
    return mnp.transpose(a, axes)

def swapaxes(data, axis0, axis1):
    return mnp.swapaxes(data, axis0, axis1)

def scatter():
    pass

def scatter_add():
    pass

def squeeze(data, axis=None):
    return mnp.squeeze(data, axis)

def expand_dims(data, axis):
    return mnp.expand_dims(data, axis)

def t(data):
    validator.check_equal_int(data.ndim, 2)
    return mnp.swapaxes(data, 1, 0)

def stack(data, axis=0):
    return mnp.stack(data, axis)

def where(condition, x=None, y=None):
    return mnp.where(condition, x, y)

@constexpr
def randn(*shape, dtype=mstype.float32):
    return Tensor(np.random.randn(*shape), dtype)

@constexpr
def normal(mean, std, shape):
    if isinstance(mean, Tensor):
        mean = mean.asnumpy()
    if isinstance(std, Tensor):
        mean = std.asnumpy()
    return Tensor(np.random.normal(mean, std, shape))