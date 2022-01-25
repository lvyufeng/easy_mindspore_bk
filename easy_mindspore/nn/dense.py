import math
import mindspore
from mindspore.common.parameter import Parameter
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, HeUniform, Uniform, _calculate_fan_in_and_fan_out

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

class BiDense(nn.Cell):
    def __init__(self, in1_channels: int, in2_channels: int, out_channels:int, \
        has_bias: bool = True):
        super().__init__()
        self.in1_channels = in1_channels
        self.in2_channels = in2_channels
        self.out_channels = out_channels
        self.has_bias = has_bias
        bound = 1 / math.sqrt(in1_channels)
        self.weight = Parameter(initializer(Uniform(bound), (out_channels, in1_channels, in2_channels)), 'weight')
        if self.has_bias:
            self.bias = Parameter(initializer(Uniform(bound), (out_channels,)), 'bias')
        else:
            self.bias = None

    def construct(self, input1, input2):
        batch_size = input1.shape[0]
        # (batch, in1) * (in1, in2, out) = (batch, in2, out)
        output = ops.matmul(input1, self.weight.transpose(1, 2, 0).view(self.in1_channels, -1))
        output = output.view(batch_size, self.in2_channels, self.out_channels)
        # (out, batch, in2) * (batch, in2) = (out, batch, in2)
        output = output.transpose(2, 0, 1) * input2
        # (out, batch, in2) -> (batch, out)
        output = output.sum(2).swapaxes(0, 1)
        if self.has_bias:
            output += self.bias
        return output
