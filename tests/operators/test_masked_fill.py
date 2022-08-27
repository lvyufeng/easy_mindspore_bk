import unittest
import mindspore
from mindspore import Tensor
from easy_mindspore.func.custom import MaskedFill
import mindspore.ops as P

class TestMaskedFill(unittest.TestCase):
    def test_float_mask(self):
        net = MaskedFill(123.0)
        inputs = Tensor([1., 2., 3.], mindspore.float32)
        mask = Tensor([0., 0., 1.], mindspore.float32)
        outputs = net(inputs, mask)

        assert all(P.Equal()(outputs, Tensor([1., 2., 123.], mindspore.float32)))

    def test_boolean_mask(self):
        net = MaskedFill(123.0)
        inputs = Tensor([1., 2., 3.], mindspore.float32)
        mask = Tensor([False, False, True], mindspore.float32)
        outputs = net(inputs, mask)

        assert all(P.Equal()(outputs, Tensor([1., 2., 123.], mindspore.float32)))


