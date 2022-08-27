import math
from typing import Tuple, Union
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.initializer import initializer, HeUniform, Uniform,_calculate_fan_in_and_fan_out

class _ConvNd(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        groups: int,
        has_bias: bool,
        padding_mode: str,
        weight_init,
        bias_init
    ):
        super().__init__()
        valid_padding_modes = {'zeros', 'reflect'}
        assert padding_mode in valid_padding_modes
        self.padding_mode = padding_mode
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.has_bias = has_bias
        self.weight = Parameter(initializer(self.weight_init, \
            [in_channels, out_channels // groups, *kernel_size]), 'weight')
        if self.has_bias:
            self.bias = Parameter(initializer(self.bias_init, [out_channels]), 'bias')
        else:
            self.bias = None

    def construct(self, inputs):
        raise NotImplementedError

class Conv1d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[str, int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        has_bias: bool = True,
        padding_mode: str = 'zeros',
    ):
        fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
        bound = 1 / math.sqrt(fan_in)
        weight_init = HeUniform(math.sqrt(5))
        bias_init = Uniform(bound)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
            has_bias, padding_mode, weight_init, bias_init
        )
        self.conv = ops.Conv2D(
            out_channels, kernel_size, 
        )
    def construct(self, inputs):
        if self.padding_mode != 'zeros':
            inputs = ops.MirrorPad()(inputs)
        return super().construct(inputs)