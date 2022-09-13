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
        x = ems.randn(4)
        y = ems.randn(2, 3)
        assert x.shape == (4,)
        assert y.shape == (2, 3)

    def test_randint(self):
        x = ems.randint(3, 5, (3,))
        y = ems.randint(3, 10, (2, 2))
        print(x, y)
        assert x.shape == (3,)
        assert y.shape == (2, 2)