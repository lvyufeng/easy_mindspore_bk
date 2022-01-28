import unittest
import numpy as np
import torch   
import mindspore
from easy_mindspore.nn.functional import cross_entropy as cross_entropy_ms
from torch.nn.functional import cross_entropy as cross_entropy_pt

class TestCrossEntropy(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = np.random.randn(3, 5)
        self.target = np.array([1, 0, 4])
        self.weight = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        return super().setUp()

    def test_cross_entropy_mean(self):
        inputs_ms = mindspore.Tensor(self.inputs, mindspore.float32)
        target_ms = mindspore.Tensor(self.target, mindspore.int32)

        res_ms = cross_entropy_ms(inputs_ms, target_ms)

        inputs_pt = torch.tensor(self.inputs)
        target_pt = torch.tensor(self.target)

        res_pt = cross_entropy_pt(inputs_pt, target_pt)

        assert np.allclose(res_ms.asnumpy(), res_pt.numpy(), 1e-3, 1e-3)
    
    def test_cross_entropy_sum_with_weight(self):
        inputs_ms = mindspore.Tensor(self.inputs, mindspore.float32)
        target_ms = mindspore.Tensor(self.target, mindspore.int32)
        
        weight_ms = mindspore.Tensor(self.weight, mindspore.float32)
        res_ms = cross_entropy_ms(inputs_ms, target_ms, reduction='sum', weight=weight_ms)

        inputs_pt = torch.tensor(self.inputs)
        target_pt = torch.tensor(self.target)
        weight_pt = torch.tensor(self.weight)

        res_pt = cross_entropy_pt(inputs_pt, target_pt, reduction='sum', weight=weight_pt)

        assert np.allclose(res_ms.asnumpy(), res_pt.numpy(), 1e-3, 1e-3)

    def test_cross_entropy_mean_label_smoothing(self):
        inputs_ms = mindspore.Tensor(self.inputs, mindspore.float32)
        target_ms = mindspore.Tensor(self.target, mindspore.int32)

        res_ms = cross_entropy_ms(inputs_ms, target_ms, label_smoothing=0.1)

        inputs_pt = torch.tensor(self.inputs)
        target_pt = torch.tensor(self.target)

        res_pt = cross_entropy_pt(inputs_pt, target_pt, label_smoothing=0.1)

        assert np.allclose(res_ms.asnumpy(), res_pt.numpy(), 1e-3, 1e-3)
    