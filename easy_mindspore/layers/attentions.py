import mindspore.nn as nn
from mindspore import Tensor

class AdditiveAttention(nn.Cell):
    pass

class DotAttention(nn.Cell):
    pass

class BiAttention(nn.Cell):
    pass

class CosineAttention(nn.Cell):
    pass

class ScaledDotProductAttention(nn.Cell):
    pass

class SelfAttention(nn.Cell):
    pass

class MultiHeadAttention(nn.Cell):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=False,
                 kdim=None,
                 vdim=None,
                 batch_first=False,
                 dtyep=None
                ):
        super(MultiHeadAttention, self).__init__()

    def construct(self, query:Tensor, key:Tensor, value:Tensor):
        pass


