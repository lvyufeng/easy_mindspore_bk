import unittest
import mindspore
import mindspore.numpy as mnp
from mindspore import Tensor
from easy_mindspore.nn import ReflectionPad1d, ReflectionPad2d, ReflectionPad3d

class TestReflectionPad(unittest.TestCase):
    def test_reflection_pad_1d(self):
        m = ReflectionPad1d(2)
        inputs = mnp.arange(8).reshape(1, 2, 4)
        outputs = m(inputs)

        expected = Tensor([[[2, 1, 0, 1, 2, 3, 2, 1],
                            [6, 5, 4, 5, 6, 7, 6, 5]]])
        assert mnp.array_equal(outputs, expected)

    def test_reflection_pad_2d(self):
        m = ReflectionPad2d(2)
        inputs = mnp.arange(9).reshape(1, 1, 3, 3)
        outputs = m(inputs)

        expected = Tensor([[[[8, 7, 6, 7, 8, 7, 6],
                             [5, 4, 3, 4, 5, 4, 3],
                             [2, 1, 0, 1, 2, 1, 0],
                             [5, 4, 3, 4, 5, 4, 3],
                             [8, 7, 6, 7, 8, 7, 6],
                             [5, 4, 3, 4, 5, 4, 3],
                             [2, 1, 0, 1, 2, 1, 0]]]])
        assert mnp.array_equal(outputs, expected)

    def test_reflection_pad_3d(self):
        m = ReflectionPad3d(1)
        inputs = mnp.arange(8, dtype=mindspore.float32).reshape(1, 1, 2, 2, 2)
        print(inputs)
        outputs = m(inputs)
        print(outputs)
        expected = Tensor([[[[[7., 6., 7., 6.],
                              [5., 4., 5., 4.],
                              [7., 6., 7., 6.],
                              [5., 4., 5., 4.]],
                             [[3., 2., 3., 2.],
                              [1., 0., 1., 0.],
                              [3., 2., 3., 2.],
                              [1., 0., 1., 0.]],
                             [[7., 6., 7., 6.],
                              [5., 4., 5., 4.],
                              [7., 6., 7., 6.],
                              [5., 4., 5., 4.]],
                             [[3., 2., 3., 2.],
                              [1., 0., 1., 0.],
                              [3., 2., 3., 2.],
                              [1., 0., 1., 0.]]]]], mindspore.float32)
        assert mnp.array_equal(outputs, expected)
