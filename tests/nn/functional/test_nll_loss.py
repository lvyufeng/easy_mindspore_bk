import unittest
import numpy as np
import torch   
import mindspore
import mindspore.numpy as mnp
from easy_mindspore.nn.functional import nll_loss as nll_loss_ms
from torch.nn.functional import nll_loss as nll_loss_pt

class TestNLLLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = np.random.randn(3, 5)
        self.target = np.array([1, 0, 4])
        self.weight = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        return super().setUp()

    def test_nll_loss_mean(self):
        inputs_ms = mindspore.Tensor(self.inputs, mindspore.float32)
        target_ms = mindspore.Tensor(self.target, mindspore.int32)

        res_ms = nll_loss_ms(inputs_ms, target_ms)

        inputs_pt = torch.tensor(self.inputs)
        target_pt = torch.tensor(self.target)

        res_pt = nll_loss_pt(inputs_pt, target_pt)

        assert np.allclose(res_ms.asnumpy(), res_pt.numpy(), 1e-3, 1e-3)

    def test_nll_loss_sum(self):
        inputs_ms = mindspore.Tensor(self.inputs, mindspore.float32)
        target_ms = mindspore.Tensor(self.target, mindspore.int32)

        res_ms = nll_loss_ms(inputs_ms, target_ms, reduction='sum')

        inputs_pt = torch.tensor(self.inputs)
        target_pt = torch.tensor(self.target)

        res_pt = nll_loss_pt(inputs_pt, target_pt, reduction='sum')

        assert np.allclose(res_ms.asnumpy(), res_pt.numpy(), 1e-3, 1e-3)

    def test_nll_loss_sum_with_weight(self):
        inputs_ms = mindspore.Tensor(self.inputs, mindspore.float32)
        target_ms = mindspore.Tensor(self.target, mindspore.int32)
        
        weight_ms = mindspore.Tensor(self.weight, mindspore.float32)
        res_ms = nll_loss_ms(inputs_ms, target_ms, reduction='none', weight=weight_ms)

        inputs_pt = torch.tensor(self.inputs)
        target_pt = torch.tensor(self.target)
        weight_pt = torch.tensor(self.weight)

        res_pt = nll_loss_pt(inputs_pt, target_pt, reduction='none', weight=weight_pt)
        print(res_ms, res_pt)

        assert np.allclose(res_ms.asnumpy(), res_pt.numpy(), 1e-3, 1e-3)

    def test_nll_loss_mean_ignore_index(self):
        inputs_ms = mindspore.Tensor(self.inputs, mindspore.float32)
        target_ms = mindspore.Tensor(self.target, mindspore.int32)

        res_ms = nll_loss_ms(inputs_ms, target_ms, ignore_index=0)

        inputs_pt = torch.tensor(self.inputs)
        target_pt = torch.tensor(self.target)

        res_pt = nll_loss_pt(inputs_pt, target_pt, ignore_index=0)

        assert np.allclose(res_ms.asnumpy(), res_pt.numpy(), 1e-3, 1e-3)