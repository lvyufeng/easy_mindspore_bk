import numpy as np
from mindspore.common.initializer import Initializer, _assignment

class Uniform(Initializer):
    r"""
    Generates an array with values sampled from Uniform distribution :math:`{U}(-\text{scale}, \text{scale})` in order
    to initialize a tensor.

    Args:
        scale (float): The bound of the Uniform distribution. Default: 0.07.


    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, Uniform
        >>> tensor1 = initializer(Uniform(), [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer('uniform', [1, 2, 3], mindspore.float32)
    """
    def __init__(self, low, high):
        super(Uniform, self).__init__()
        self.low = low
        self.high = high

    def _initialize(self, arr):
        tmp = np.random.uniform(self.low, self.high, arr.shape)
        _assignment(arr, tmp)