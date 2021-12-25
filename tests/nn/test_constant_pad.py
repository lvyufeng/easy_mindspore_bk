import unittest
import mindspore
import mindspore.numpy as mnp
from mindspore import Tensor
from easy_mindspore.nn.padding_layers import ConstantPad1d

class TestConstantPad(unittest.TestCase):
    def test_constant_pad_1d(self):
        m = ConstantPad1d(2, 3.5)
        inputs = mnp.arange(8, dtype=mindspore.float32).reshape(1, 2, 4)
        outputs = m(inputs)
        print(outputs)
        expected = Tensor([[[3.5, 3.5, 0., 1., 2., 3., 3.5, 3.5],
                            [3.5, 3.5, 4., 5., 6., 7., 3.5, 3.5]]], mindspore.float32)
        assert mnp.array_equal(outputs, expected)
        m = ConstantPad1d((3, 1), 3.5)
        outputs = m(inputs)

        expected = Tensor([[[3.5, 3.5, 3.5, 0., 1., 2., 3., 3.5],
                            [3.5, 3.5, 3.5, 4., 5., 6., 7., 3.5]]], mindspore.float32)
        assert mnp.array_equal(outputs, expected)