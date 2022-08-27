import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Uniform
from .dense import Dense

class AdditiveAttention(nn.Cell):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj = Dense(hidden_dim, hidden_dim, has_bias=False)
        self.key_proj = Dense(hidden_dim, hidden_dim, has_bias=False)
        self.bias = Parameter(initializer(Uniform(0.1), hidden_dim), 'bias')
        self.score_proj = Dense(hidden_dim, 1)

    def construct(self, query, key, value):
        score = self.score_proj(ops.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = ops.Softmax()(score)
        context = ops.matmul(attn.expand_dims(1), value)
        return context, attn

class DotAttention(nn.Cell):
    pass

class BiAttention(nn.Cell):
    pass

class CosineAttention(nn.Cell):
    pass

class ScaledDotProductAttention(nn.Cell):
    def __init__(self, dim):
        super().__init__()


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


