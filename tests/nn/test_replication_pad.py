import unittest
import mindspore
import mindspore.numpy as mnp
from mindspore import Tensor
from easy_mindspore.nn.padding_layers import ReplicationPad1d

class TestReplicationPad(unittest.TestCase):
    def test_replication_pad_1d(self):
        m = ReplicationPad1d(2)
        inputs = mnp.arange(8, dtype=mindspore.float32).reshape(1, 2, 4)
        outputs = m(inputs)

        expected = Tensor([[[0., 0., 0., 1., 2., 3., 3., 3.],
                            [4., 4., 4., 5., 6., 7., 7., 7.]]], mindspore.float32)
        assert mnp.array_equal(outputs, expected)
        m = ReplicationPad1d((3, 1))
        outputs = m(inputs)

        expected = Tensor([[[0., 0., 0., 0., 1., 2., 3., 3.],
                            [4., 4., 4., 4., 5., 6., 7., 7.]]], mindspore.float32)
        assert mnp.array_equal(outputs, expected)