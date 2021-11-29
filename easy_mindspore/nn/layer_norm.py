import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.initializer import initializer

class LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, gamma_init='ones', beta_init='zeros', epsilon=1e-05):
        """Initialize LayerNorm."""
        super(LayerNorm, self).__init__()
        if not isinstance(normalized_shape, (tuple, list)):
            raise TypeError(f"For '{self.cls_name}', the type of 'normalized_shape' should be tuple[int] or list[int], "
                            f"but got {normalized_shape} and the type is {type(normalized_shape)}.")
        self.normalized_shape = normalized_shape
        self.norm_ndim = len(normalized_shape)
        self.epsilon = epsilon
        self.gamma = Parameter(initializer(
            gamma_init, normalized_shape), name="gamma")
        self.beta = Parameter(initializer(
            beta_init, normalized_shape), name="beta")

    def construct(self, input_x):
        layer_norm = ops.LayerNorm(begin_norm_axis=input_x.ndim - self.norm_ndim,
                              begin_params_axis=input_x.ndim - self.norm_ndim,
                              epsilon=self.epsilon)
        y, _, _ = layer_norm(input_x, self.gamma, self.beta)
        return y