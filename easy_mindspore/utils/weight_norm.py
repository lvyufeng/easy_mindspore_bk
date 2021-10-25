from mindspore.common.parameter import Parameter
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore.ops import constexpr

def norm_except_dim(v, pow, dim):
    if dim == -1:
        return mnp.norm(v, pow)
    elif dim == 0:
        output_size = (v.shape[0],) + (1,) * (v.ndim - 1)
        return mnp.norm(v.view((v.shape[0], -1)), pow, 1).view(output_size)
    elif dim == (v.ndim - 1):
        output_size = (1,) * (v.ndim - 1) + (v.shape[v.ndim - 1])
        return mnp.norm(v.view((-1, v.shape[v.ndim - 1])), pow, 0).view(output_size)
    else:
        return norm_except_dim(v.swapaxes(0, dim), pow, dim).swapaxes(0, dim)

def _weight_norm(v, g, dim):
    return v * (g / norm_except_dim(v, 2, dim))

class WeightNorm(nn.Cell):
    r"""Applies weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. 
    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (Module): containing module
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = WeightNorm(nn.Dense(20, 40))
        >>> m.param_g.shape
        (40, 1)
        >>> m.param_v.shape
        (40, 20)

    """
    def __init__(self, module, dim:int=0):
        super().__init__()
        if dim is None:
            dim = -1
        self.dim = dim
        self.module = module
        self.assign = P.Assign()
        # add g and v as new parameters and express w as g/||v|| * v
        self.param_g = Parameter(Tensor(norm_except_dim(self.module.weight, 2, dim)))
        self.param_v = Parameter(Tensor(self.module.weight.data))
        self.module.weight.set_data(_weight_norm(self.param_v, self.param_g, self.dim))
        self.use_weight_norm = True

    def construct(self, *inputs, **kwargs):
        if not self.use_weight_norm:
            return self.module(*inputs, **kwargs)
        self.assign(self.module.weight, _weight_norm(self.param_v, self.param_g, self.dim))
        return self.module(*inputs, **kwargs)

    def remove_weight_norm(self):
        self.assign(self.module.weight, _weight_norm(self.param_v, self.param_g, self.dim))
        self.use_weight_norm = False

