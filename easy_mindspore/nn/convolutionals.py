import math
import mindspore.nn as nn
from mindspore.common.initializer import initializer, HeUniform, Uniform,_calculate_fan_in_and_fan_out

class Conv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, pad_mode, padding, dilation, group, has_bias, weight_init='normal', bias_init='zeros')
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))