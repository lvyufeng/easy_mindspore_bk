from cmath import log
import unittest
import numpy as np
import torch   
import mindspore
import mindspore.numpy as mnp
from easy_mindspore.nn.functional import kl_div as kl_div_ms
from torch.nn.functional import kl_div as kl_div_pt

class TestKLDiv(unittest.TestCase):
    def setUp(self) -> None:
        self.p = [0.4, 0.4, 0.2]
        self.q = [0.5, 0.1, 0.4]
        return super().setUp()

    def test_kl_div_sum(self):
        p_ms = mindspore.Tensor(self.p, mindspore.float32)
        q_ms = mindspore.Tensor(self.q, mindspore.float32)

        res_ms = kl_div_ms(mnp.log(q_ms), p_ms, 'sum')

        p_pt = torch.tensor(self.p)
        q_pt = torch.tensor(self.q)
        res_pt = kl_div_pt(q_pt.log(), p_pt, reduction='sum')

        assert np.allclose(res_ms.asnumpy(), res_pt.numpy(), 1e-3, 1e-3)

    def test_kl_div_mean(self):
        p_ms = mindspore.Tensor(self.p, mindspore.float32)
        q_ms = mindspore.Tensor(self.q, mindspore.float32)

        res_ms = kl_div_ms(mnp.log(q_ms), p_ms, 'mean')

        p_pt = torch.tensor(self.p)
        q_pt = torch.tensor(self.q)
        res_pt = kl_div_pt(q_pt.log(), p_pt, reduction='mean')

        assert np.allclose(res_ms.asnumpy(), res_pt.numpy(), 1e-3, 1e-3)

    def test_kl_div_sum_log_target(self):
        p_ms = mindspore.Tensor(self.p, mindspore.float32)
        q_ms = mindspore.Tensor(self.q, mindspore.float32)

        res_ms = kl_div_ms(mnp.log(q_ms), mnp.log(p_ms), 'sum', log_target=True)

        p_pt = torch.tensor(self.p)
        q_pt = torch.tensor(self.q)
        res_pt = kl_div_pt(q_pt.log(), p_pt.log(), reduction='sum', log_target=True)

        assert np.allclose(res_ms.asnumpy(), res_pt.numpy(), 1e-3, 1e-3)
