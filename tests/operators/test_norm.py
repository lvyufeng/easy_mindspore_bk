import unittest
import mindspore
import numpy as np
import mindspore.ops as ops
from easy_mindspore.func import norm, inf

class TestNorm(unittest.TestCase):
    def setUp(self):
        self.a = ops.arange(9) - 4
        self.b = self.a.reshape((3, 3))
        self.c = mindspore.Tensor([[ 1, 2, 3],
                                   [-1, 1, 4]], mindspore.int32)
        self.m = ops.arange(8).reshape(2,2,2)

    def test_norm_a(self):
        out = norm(self.a)
        out_np = np.linalg.norm(self.a.asnumpy())

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_b(self):
        out = norm(self.b)
        out_np = np.linalg.norm(self.b.asnumpy())

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_b_fro(self):
        out = norm(self.b, 'fro')
        out_np = np.linalg.norm(self.b.asnumpy(), 'fro')

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_a_inf(self):
        out = norm(self.a, inf)
        out_np = np.linalg.norm(self.a.asnumpy(), np.inf)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_b_inf(self):
        out = norm(self.b, inf)
        out_np = np.linalg.norm(self.b.asnumpy(), np.inf)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_a_neg_inf(self):
        out = norm(self.a, -inf)
        out_np = np.linalg.norm(self.a.asnumpy(), -np.inf)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_b_neg_inf(self):
        out = norm(self.b, -inf)
        out_np = np.linalg.norm(self.b.asnumpy(), -np.inf)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_a_1(self):
        out = norm(self.a, 1)
        out_np = np.linalg.norm(self.a.asnumpy(), 1)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_b_1(self):
        out = norm(self.b, 1)
        out_np = np.linalg.norm(self.b.asnumpy(), 1)
        print(out, out_np)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_a_neg_1(self):
        out = norm(self.a, -1)
        out_np = np.linalg.norm(self.a.asnumpy(), -1)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_b_neg_1(self):
        out = norm(self.b, -1)
        out_np = np.linalg.norm(self.b.asnumpy(), -1)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_a_2(self):
        out = norm(self.a, 2)
        out_np = np.linalg.norm(self.a.asnumpy(), 2)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_b_2(self):
        out = norm(self.b, 2)
        out_np = np.linalg.norm(self.b.asnumpy(), 2)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_a_neg_2(self):
        out = norm(self.a, -2)
        out_np = np.linalg.norm(self.a.asnumpy(), -2)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_b_neg_2(self):
        out = norm(self.b, -2)
        out_np = np.linalg.norm(self.b.asnumpy(), -2)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_a_3(self):
        out = norm(self.a, 3)
        out_np = np.linalg.norm(self.a.asnumpy(), 3)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_a_neg_3(self):
        out = norm(self.a, -3)
        out_np = np.linalg.norm(self.a.asnumpy(), -3)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_a_axis_0(self):
        out = norm(self.c, axis=0)
        out_np = np.linalg.norm(self.c.asnumpy(), axis=0)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_a_axis_1(self):
        out = norm(self.c, axis=1)
        out_np = np.linalg.norm(self.c.asnumpy(), axis=1)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_a_ord_1_axis_1(self):
        out = norm(self.c, ord=1, axis=1)
        out_np = np.linalg.norm(self.c.asnumpy(), ord=1, axis=1)

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_a_axis_1_2(self):
        out = norm(self.m, axis=(1,2))
        out_np = np.linalg.norm(self.m.asnumpy(), axis=(1,2))

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_a_slice_0(self):
        out = norm(self.m[0:, :, :])
        out_np = np.linalg.norm(self.m[0:, :, :].asnumpy())

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)

    def test_norm_a_slice_1(self):
        out = norm(self.m[1:, :, :])
        out_np = np.linalg.norm(self.m[1:, :, :].asnumpy())

        assert np.allclose(out.asnumpy(), out_np, 1e-5, 1e-5)