"""include missing operators and some new layers like Transformer."""
from mindspore.nn.layer.activation import Sigmoid
from .convolutionals import Conv1d
from .pooling_layers import *
# dense layers
from .dense import *
# embedding layers
from .embeddings import Embedding
# padding layers
from .padding_layers import *
# normalizations
from .norm_layers import LayerNorm
# activations
from mindspore.nn import ELU, HShrink, HSigmoid, HSwish, \
    LeakyReLU, LogSigmoid, PReLU, ReLU, ReLU6, GELU, \
    Sigmoid, SoftShrink, Tanh, LogSoftmax
from .activations import *
# rnns
from mindspore.nn import LSTM, GRU, RNN, \
    LSTMCell, GRUCell, RNNCell
