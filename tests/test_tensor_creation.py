import unittest
import mindspore
import mindspore.common.dtype as mstype
import easy_mindspore as ems

class TestTensorCreation(unittest.TestCase):
    def test_type_inference_float(self):
        x = ems.tensor([1., 2., 3.])
        y = ems.tensor([1, 2, 3])
        assert x.dtype == mstype.float32
        assert y.dtype == mstype.int32

    def test_randn(self):
        x = ems.randn(4, dtype=mindspore.float32)
        y = ems.randn(2, 3, dtype=mindspore.float32)
        assert x.shape == (4,)
        assert y.shape == (2, 3)
