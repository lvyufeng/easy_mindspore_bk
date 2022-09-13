import mindspore
from mindspore import Tensor
from mindspore.common.initializer import Normal
from ..initializer import Uniform
# bernoulli
# multinomial
# normal
# poisson
# rand
# rand_like
# randint
def randint(low, high, size, dtype=None):
    if dtype is None:
        dtype = mindspore.int32
    return Tensor(shape=size, dtype=dtype, init=Uniform(low, high))
# randint_like
# randn
def randn(*size, dtype=None):
    if dtype is None:
        dtype = mindspore.float32
    return Tensor(shape=size, dtype=dtype, init=Normal(1.0))
# randn_like
# randperm