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
        tensors = ()
        for k in range(self.out_channels):
            buff = ops.MatMul()(input1, self.weight[k])
            buff = buff * input2
            tensors += (buff.sum(1, keepdims=True), )
        outputs = ops.Concat(1)(tensors)
        if self.has_bias:
            outputs += self.bias
        return outputs