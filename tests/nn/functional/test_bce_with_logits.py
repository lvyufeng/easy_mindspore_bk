import unittest
import numpy as np
import torch   
import mindspore
from easy_mindspore.nn.functional import binary_cross_entropy_with_logits as binary_cross_entropy_with_logits_ms
from torch.nn.functional import binary_cross_entropy_with_logits as binary_cross_entropy_with_logits_pt

class TestCrossEntropy(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = np.random.randn(3)
        self.target = np.array([1., 0., 1.])
        self.weight = np.array([0.1, 0.2, 0.3])
        self.pos_weight = np.array([0.8, 0.2, 1.0])
        return super().setUp()

    def test_binary_cross_entropy_with_logits_mean(self):
        inputs_ms = mindspore.Tensor(self.inputs, mindspore.float32)
        target_ms = mindspore.Tensor(self.target, mindspore.int32)

        res_ms = binary_cross_entropy_with_logits_ms(inputs_ms, target_ms)

        inputs_pt = torch.tensor(self.inputs)
        target_pt = torch.tensor(self.target)

        res_pt = binary_cross_entropy_with_logits_pt(inputs_pt, target_pt)

        assert np.allclose(res_ms.asnumpy(), res_pt.numpy(), 1e-3, 1e-3)
    
    def test_binary_cross_entropy_with_logits_sum_with_weight(self):
        inputs_ms = mindspore.Tensor(self.inputs, mindspore.float32)
        target_ms = mindspore.Tensor(self.target, mindspore.int32)
        
        weight_ms = mindspore.Tensor(self.weight, mindspore.float32)
        res_ms = binary_cross_entropy_with_logits_ms(inputs_ms, target_ms, reduction='sum', weight=weight_ms)

        inputs_pt = torch.tensor(self.inputs)
        target_pt = torch.tensor(self.target)
        weight_pt = torch.tensor(self.weight)

        res_pt = binary_cross_entropy_with_logits_pt(inputs_pt, target_pt, reduction='sum', weight=weight_pt)

        assert np.allclose(res_ms.asnumpy(), res_pt.numpy(), 1e-3, 1e-3)

    def test_binary_cross_entropy_with_logits_mean_with_pos_weight(self):
        inputs_ms = mindspore.Tensor(self.inputs, mindspore.float32)
        target_ms = mindspore.Tensor(self.target, mindspore.int32)
        weight_ms = mindspore.Tensor(self.weight, mindspore.float32)
        pos_weight_ms = mindspore.Tensor(self.pos_weight, mindspore.float32)
        res_ms = binary_cross_entropy_with_logits_ms(inputs_ms, target_ms, reduction='mean', weight=weight_ms, pos_weight=pos_weight_ms)

        inputs_pt = torch.tensor(self.inputs)
        target_pt = torch.tensor(self.target)
        weight_pt = torch.tensor(self.weight)
        pos_weight_pt = torch.tensor(self.pos_weight)

        res_pt = binary_cross_entropy_with_logits_pt(inputs_pt, target_pt, reduction='mean', weight=weight_pt, pos_weight=pos_weight_pt)

        assert np.allclose(res_ms.asnumpy(), res_pt.numpy(), 1e-3, 1e-3)
