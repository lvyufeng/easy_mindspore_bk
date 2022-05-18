from math import tanh
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.seed import _get_graph_seed
from mindspore import Tensor
from torch import logit

class CELU(nn.Cell):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.exp = ops.Exp()
        self.max = ops.Maximum()
        self,min = ops.Minimum()

    def construct(self, inputs):
        return self.max(0, inputs) + \
            self.min(0, self.alpha * (self.exp(inputs / self.alpha) - 1))

class HTanh(nn.Cell):
    r"""Applies the HardTanh function element-wise

    HardTanh is defined as:

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            1 & \text{ if } x > 1 \\
            -1 & \text{ if } x < -1 \\
            x & \text{ otherwise } \\
        \end{cases}

    The range of the linear region :math:`[-1, 1]` can be adjusted using
    :attr:`min_val` and :attr:`max_val`.

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1

    Keyword arguments :attr:`min_value` and :attr:`max_value`
    have been deprecated in favor of :attr:`min_val` and :attr:`max_val`.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> import mindspore
        >>> m = nn.HTanh(-2, 2)
        >>> inputs = mindspore.Tensor([1, 2, 3], mindspore.float32)
        >>> outputs = m(inputs)
    """
    def __init__(self, min_val=-1.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.max = ops.Maximum()
        self.min = ops.Minimum()

    def construct(self, inputs):
        x_min = self.min(inputs, self.max_val)
        x_max = self.max(x_min, self.min_val)
        return x_max

class RReLU(nn.Cell):
    r"""Applies the randomized leaky rectified liner unit function, element-wise,
    as described in the paper:

    `Empirical Evaluation of Rectified Activations in Convolutional Network`_.

    The function is defined as:

    .. math::
        \text{RReLU}(x) =
        \begin{cases}
            x & \text{if } x \geq 0 \\
            ax & \text{ otherwise }
        \end{cases}

    where :math:`a` is randomly sampled from uniform distribution
    :math:`\mathcal{U}(\text{lower}, \text{upper})`.

     See: https://arxiv.org/pdf/1505.00853.pdf

    Args:
        lower: lower bound of the uniform distribution. Default: :math:`\frac{1}{8}`
        upper: upper bound of the uniform distribution. Default: :math:`\frac{1}{3}`
        seed: The random sample seed. Default: 0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.RReLU(0.1, 0.3)
        >>> inputs = mindspore.Tensor([1, 2, 3], mindspore.float32)
        >>> outputs = m(inputs)

    .. _`Empirical Evaluation of Rectified Activations in Convolutional Network`:
        https://arxiv.org/abs/1505.00853
    """
    def __init__(
        self,
        lower: float = 1. / 8,
        upper: float = 1. / 3,
        seed=0
    ):
        super().__init__()
        seed1, seed2 = _get_graph_seed(seed, 'uniform')
        self.uniform = ops.UniformReal(seed1, seed2)
        self.lower = Tensor(lower, mstype.float32)
        self.upper = Tensor(upper, mstype.float32)
        self.relu = ops.ReLU()

    def construct(self, inputs):
        if self.training:
            alpha = self.uniform(inputs.shape, self.lower, self.upper)
        else:
            alpha = (self.upper + self.lower) / 2
        return self.relu(inputs) - self.relu(-inputs) * alpha
        
class SELU(nn.Cell):
    r"""Applied element-wise, as:

    .. math::
        \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))

    with :math:`\alpha = 1.6732632423543772848170429916717` and
    :math:`\text{scale} = 1.0507009873554804934193349852946`.

    More details can be found in the paper `Self-Normalizing Neural Networks`_ .

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SELU()
        >>> inputs = mindspore.Tensor([1, 2, 3], mindspore.float32)
        >>> outputs = m(inputs)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """
    def __init__(self):
        super().__init__()
        self.selu = ops.SeLU()
    
    def construct(self, inputs):
        return self.selu(inputs)

class SiLU(nn.Cell):
    r"""Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SiLU()
        >>> inputs = mindspore.Tensor([1, 2, 3], mindspore.float32)
        >>> outputs = m(inputs)
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = ops.Sigmoid()

    def construct(self, inputs):
        return inputs * self.sigmoid(inputs)

class Mish(nn.Cell):
    r"""Applies the Mish function, element-wise.
    Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    .. math::
        \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

    .. note::
        See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Mish()
        >>> inputs = mindspore.Tensor([1, 2, 3], mindspore.float32)
        >>> outputs = m(inputs)
    """
    def __init__(self):
        super().__init__()
        self.mish = ops.Mish()
    
    def construct(self, inputs):
        return self.mish(inputs)

class Softplus(nn.Cell):
    r"""Applies the element-wise function:

    .. math::
        \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))

    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.

    For numerical stability the implementation reverts to the linear function
    when :math:`input \times \beta > threshold`.

    Args:
        beta: the :math:`\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Softplus()
        >>> inputs = mindspore.Tensor([1, 2, 3], mindspore.float32)
        >>> outputs = m(inputs)
    """
    def __init__(self):
        super().__init__()
        self.softplus = ops.Softplus()

    def construct(self, inputs):
        return self.softplus(inputs)

class Softsign(nn.Cell):
    r"""Applies the element-wise function:

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Softsign()
        >>> inputs = mindspore.Tensor([1, 2, 3], mindspore.float32)
        >>> outputs = m(inputs)
    """
    def __init__(self):
        super().__init__()
        self.softsign = ops.Softsign()
    
    def construct(self, inputs):
        return self.softsign(inputs)

class Tanhshrink(nn.Cell):
    r"""Applies the element-wise function:

    .. math::
        \text{Tanhshrink}(x) = x - \tanh(x)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Tanhshrink()
        >>> inputs = mindspore.Tensor([1, 2, 3], mindspore.float32)
        >>> outputs = m(inputs)
    """
    def __init__(self):
        super().__init__()
        self.tanh = ops.Tanh()

    def construct(self, inputs):
        return inputs - self.tanh(inputs)

class Threshold(nn.Cell):
    r"""Thresholds each element of the input Tensor.

    Threshold is defined as:

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    Args:
        threshold: The value to threshold at
        value: The value to replace with

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Threshold(0.1, 20)
        >>> inputs = mindspore.Tensor([0.1, 0.2, 0.3], mindspore.float32)
        >>> outputs = m(inputs)
    """
    def __init__(self, threshold, value):
        super().__init__()
        self.threshold = threshold
        self.value = value
        self.greater = ops.Greater()
        self.fill = ops.Fill()
        self.select = ops.Select()

    def construct(self, inputs):
        cond = self.greater(inputs, self.threshold)
        value = self.fill(inputs.dtype, inputs.shape, self.value)
        return self.select(cond, inputs, value)

class GLU(nn.Cell):
    r"""Applies the gated linear unit function
    :math:`{GLU}(a, b)= a \otimes \sigma(b)` where :math:`a` is the first half
    of the input matrices and :math:`b` is the second half.

    Args:
        dim (int): the dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::

        >>> m = nn.GLU()
        >>> inputs = ops.ones(4, 2)
        >>> outputs = m(inputs)
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.split = ops.Split(dim, 2)
        self.sigmoid = ops.Sigmoid()

    def construct(self, inputs):
        a, b = self.split(inputs)
        return a * self.sigmoid(b)

class Softmin(nn.Cell):
    r"""Applies the Softmin function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range `[0, 1]` and sum to 1.

    Softmin is defined as:

    .. math::
        \text{Softmin}(x_{i}) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Args:
        axis (int): A dimension along which Softmin will be computed (so every slice
            along dim will sum to 1).

    Returns:
        a Tensor of the same dimension and shape as the input, with
        values in the range [0, 1]

    Examples::

        >>> m = nn.Softmin()
        >>> inputs = ops.ones(4, 2)
        >>> outputs = m(inputs)
    """

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis
        self.exp = ops.Exp()
        self.reduce_sum = ops.ReduceSum(keep_dims=True)

    def construct(self, inputs):
        x_exp = self.exp(-inputs)
        partion = self.reduce_sum(x_exp, self.axis)
        return x_exp / partion

class Softmax2d(nn.Cell):
    r"""Applies SoftMax over features to each spatial location.

    When given an image of ``Channels x Height x Width``, it will
    apply `Softmax` to each location :math:`(Channels, h_i, w_j)`

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        - Output: :math:`(N, C, H, W)` or :math:`(C, H, W)` (same shape as input)

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Examples::

        >>> m = nn.Softmax2d()
        >>> # you softmax over the 2nd dimension
        >>> input = ops.ones(2, 3, 12, 13)
        >>> output = m(input)
    """
    def __init__(self):
        super().__init__()
        self.exp = ops.Exp()
        self.reduce_sum = ops.ReduceSum(keep_dims=True)

    def construct(self, inputs):
        if inputs.ndim == 3:
            perm = (1, 2)
        elif inputs.ndim == 4:
            perm = (1, 2, 3)
        x_exp = self.exp(inputs)
        partion = self.reduce_sum(x_exp, perm)
        return x_exp / partion

class GumbelSoftmax(nn.Cell):
    def __init__(self, temperature=1, hard=False, axis=-1):
        super().__init__()
        self.temperature = temperature
        self.hard = hard
        self.axis = axis
        self.uniform = ops.UniformReal()
        self.softmax = ops.Softmax(axis)
        self.on_value = Tensor(1.0, mindspore.float32)
        self.off_value = Tensor(0.0, mindspore.float32)

    def construct(self, logits):
        uniform_samples = self.uniform(logits.shape)
        gumbels = -ops.log(-ops.log(uniform_samples)) # ~Gumbel(0, 1)
        gumbels = (logits + gumbels) / self.temperature
        y_soft = self.softmax(gumbels)

        if self.hard:
            # Straight through
            index = y_soft.argmax(self.axis)
            y_hard = ops.OneHot(self.axis)(index, y_soft.shape[self.axis], self.on_value, self.off_value)
            ret = ops.stop_gradient(y_hard - y_soft) + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret
