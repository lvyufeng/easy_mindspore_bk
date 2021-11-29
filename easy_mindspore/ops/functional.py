from mindspore import Tensor
import mindspore.ops as P
from .custom import *

# grad operations
def get_grads():
    pass

# math operations
def sqrt(x):
    sqrt_op = P.Sqrt()
    if isinstance(x, Tensor):
        return sqrt_op(x)
    if isinstance(x, int) or isinstance(x, float):
        return sqrt_op(P.ScalarToTensor()(x))

# tensor operations
def concat(inputs, axis=0):
    return P.Concat(axis=axis)(inputs)

def bmm(x, y, transpose_x=False, transpose_y=False):
    return P.BatchMatMul(transpose_x, transpose_y)(x, y)

def masked_fill_(inputs:Tensor, mask:Tensor, value:float):
    return MaskedFill(value)(inputs, mask)