import mindspore.nn as nn
import mindspore.ops as P

class TransformerEncoder(nn.Cell):
    pass

class TransformerDecoder(nn.Cell):
    pass

class TransformerEncoderLayer(nn.Cell):
    pass

class TransformerDecoderLayer(nn.Cell):
    pass

def _get_activation(activation):
    if activation == 'relu':
        return P.ReLU()
    elif activation == 'gelu':
        return P.GeLU()
    raise ValueError('Invalid activation operator. Activation should be relu/gelu, not {}'.format(activation))
