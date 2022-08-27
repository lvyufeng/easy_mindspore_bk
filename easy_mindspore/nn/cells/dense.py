import math
import mindspore
from mindspore.common.parameter import Parameter
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.common.initializer import initializer, HeUniform, Uniform, _calculate_fan_in_and_fan_out
from mindspore import Tensor
from mindspore.ops import constexpr

class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None):
        super().__init__(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=has_bias, activation=activation)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))

@constexpr
def check_dense_inputs_same_shape(input1, input2, prim_name=None):
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if input1[:-1] != input2[:-1]:
        raise ValueError(f"{msg_prefix} dimensions except the last of 'input1' must be same as input2, but got "
                         f"{input1} of 'input1' and {input2} of 'input2'")


class BiDense(nn.Cell):
    r"""
    The bilinear dense connected layer.

    Applies dense connected layer for two inputs. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{X_{1}}^{T} * \text{kernel} * \text{X_{2}} + \text{bias},

    where :math:`X_{1}` is the first input tensor, math:`X_{2}` is the second input tensor
    , :math:`\text{kernel}` is a weight matrix with the same data type as the :math:`X` created by the layer
    , and :math:`\text{bias}` is a bias vector with the same data type as the :math:`X` created by the layer
    (only if has_bias is True).

    Args:
        in1_channels (int): The number of channels in the input1 space.
        in2_channels (int): The number of channels in the input2 space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `x`. The values of str refer to the function `initializer`. Default: None.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as `x`. The values of str refer to the function `initializer`. Default: None.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.

    Inputs:
        - **input1** (Tensor) - Tensor of shape :math:`(*, in1\_channels)`. The `in_channels` in `Args` should be equal
          to :math:`in1\_channels` in `Inputs`.
        - **input2** (Tensor) - Tensor of shape :math:`(*, in2\_channels)`. The `in_channels` in `Args` should be equal
          to :math:`in2\_channels` in `Inputs`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Raises:
        TypeError: If `in1_channels`, `in2_channels` or `out_channels` is not an int.
        TypeError: If `has_bias` is not a bool.
        ValueError: If length of shape of `weight_init` is not equal to 3 or shape[0] of `weight_init`
                    is not equal to `out_channels` or shape[1] of `weight_init` is not equal to `in1_channels`
                    or shape[2] of `weight_init` is not equal to `in2_channels`.
        ValueError: If length of shape of `bias_init` is not equal to 1
                    or shape[0] of `bias_init` is not equal to `out_channels`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor(np.random.randn(128, 20), mindspore.float32)
        >>> x2 = Tensor(np.random.randn(128, 30), mindspore.float32)
        >>> net = nn.BiDense(20, 30, 40)
        >>> output = net(x1, x2)
        >>> print(output.shape)
        (128, 40)
    """

    def __init__(self,
                 in1_channels,
                 in2_channels,
                 out_channels,
                 weight_init=None,
                 bias_init=None,
                 has_bias=True):
        super().__init__()
        # self.in_channels = Validator.check_positive_int(in1_channels, "in1_channels", self.cls_name)
        # self.in_channels = Validator.check_positive_int(in2_channels, "in2_channels", self.cls_name)
        # self.out_channels = Validator.check_positive_int(out_channels, "out_channels", self.cls_name)
        # self.has_bias = Validator.check_bool(has_bias, "has_bias", self.cls_name)

        self.in1_channels = in1_channels
        self.in2_channels = in2_channels
        self.out_channels = out_channels
        self.has_bias = has_bias
        bound = 1 / math.sqrt(in1_channels)
        if weight_init is None:
            weight_init = Uniform(bound)
        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 3 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in1_channels or weight_init.shape[2] != in2_channels:
                raise ValueError(f"For '{self.cls_name}', weight init shape error. The ndim of 'weight_init' must "
                                 f"be equal to 3, the first dim must be equal to 'out_channels', the "
                                 f"second dim must be equal to 'in1_channels', and the third dim must be "
                                 f"equal to 'in2_channels'. But got 'weight_init': {weight_init}, "
                                 f"'out_channels': {out_channels}, 'in_channels': {in1_channels}, "
                                 f"'in2_channels': {in2_channels}")
        self.weight = Parameter(initializer(weight_init, (out_channels, in1_channels, in2_channels)), 'weight')

        if self.has_bias:
            if bias_init is None:
                bias_init = Uniform(bound)
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError(f"For '{self.cls_name}', bias init shape error. The ndim of 'bias_init' should "
                                     f"be equal to 1, and the first dim must be equal to 'out_channels'. But got "
                                     f"'bias_init': {bias_init}, 'out_channels': {out_channels}.")
            self.bias = Parameter(initializer(bias_init, [out_channels]), name="bias")
            self.bias_add = P.BiasAdd()
        self.matmul = P.MatMul()

    def construct(self, input1, input2):
        input1_shape = input1.shape
        input2_shape = input2.shape
        # check_dense_input_shape(input1_shape, self.cls_name)
        # check_dense_input_shape(input2_shape, self.cls_name)
        check_dense_inputs_same_shape(input1_shape, input2_shape, self.cls_name)
        if len(input1_shape) != 2:
            input1 = input1.reshape((-1, input1_shape[-1]))
            input2 = input2.reshape((-1, input2_shape[-1]))
        batch_size = input1.shape[0]
        output = self.matmul(input1, self.weight.transpose(1, 2, 0).view(self.in1_channels, -1))
        output = output.view(batch_size, self.in2_channels, self.out_channels)
        output = output.transpose(2, 0, 1) * input2
        output = output.sum(2).swapaxes(0, 1)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        if len(input1_shape) != 2:
            out_shape = input1_shape[:-1] + (-1,)
            output = output.reshape(out_shape)
        return output

    def extend_repr(self):
        s = 'in1_channels={}, in2_channels={}, output_channels={}'.format(
            self.in1_channels, self.in2_channels, self.out_channels)
        if self.has_bias:
            s += ', has_bias={}'.format(self.has_bias)
        return s
