import unittest
import mindspore
import mindspore.numpy as mnp
from mindspore import Tensor
from easy_mindspore.nn.padding_layers import ZeroPad2d

class TestReflectionPad(unittest.TestCase):
    def test_zero_pad_2d(self):
        m = ZeroPad2d(2)
        inputs = mnp.arange(9).reshape(1, 1, 3, 3)
        outputs = m(inputs)

        expected = Tensor([[[[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 2, 0, 0],
                             [0, 0, 3, 4, 5, 0, 0],
                             [0, 0, 6, 7, 8, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]]]])

        assert mnp.array_equal(outputs, expected)
        # using different paddings for different sides
        m = ZeroPad2d((1, 1, 2, 0))
        outputs = m(inputs)
        print(outputs)

        expected = Tensor([[[[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 1, 2, 0],
                             [0, 3, 4, 5, 0],
                             [0, 6, 7, 8, 0]]]])
        assert mnp.array_equal(outputs, expected)
