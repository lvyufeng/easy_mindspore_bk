import unittest
import mindspore
import torch
import numpy as np
from easy_mindspore.ops.functional import clip_grad_norm
from easy_mindspore import value_and_grad

class NetPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, inputs):
        return self.fc(inputs)

class NetMS(mindspore.nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc = mindspore.nn.Dense(10, 1, has_bias=True)
    
    def construct(self, inputs):
        return self.fc(inputs)

class TestClipGradNorm(unittest.TestCase):
    def setUp(self) -> None:
        weight = np.random.randn(1, 10).astype(np.float32)
        bias = np.random.randn(1).astype(np.float32)

        self.net_pt = NetPT()
        self.net_ms = NetMS()

        self.net_pt.fc.weight = torch.nn.Parameter(torch.tensor(weight), True)
        self.net_pt.fc.bias = torch.nn.Parameter(torch.tensor(bias), True)
        self.net_ms.fc.weight.set_data(mindspore.Tensor(weight))
        self.net_ms.fc.bias.set_data(mindspore.Tensor(bias))

        self.inputs = np.random.randn(4, 10).astype(np.float32)
        self.target = np.random.randn(4, 1).astype(np.float32)

        self.loss_pt = torch.nn.BCEWithLogitsLoss()
        self.loss_ms = mindspore.nn.BCEWithLogitsLoss()

    def test_cmp_forward(self):
        out_pt = self.net_pt(torch.tensor(self.inputs))
        out_ms = self.net_ms(mindspore.Tensor(self.inputs))

        assert np.allclose(out_ms.asnumpy(), out_pt.detach().numpy(), 1e-4, 1e-4)
    
    def test_clip_grad_norm(self):
        self.net_pt.train()
        self.net_pt.zero_grad()
        pt_params = self.net_pt.parameters()
        out_pt = self.net_pt(torch.tensor(self.inputs))
        loss_pt = self.loss_pt(out_pt, torch.tensor(self.target))
        loss_pt.backward()
        grad_norm_pt = torch.nn.utils.clip_grad_norm_(pt_params, 1.0)
        print(grad_norm_pt)

        net_ms = self.net_ms
        loss_ms_fn = self.loss_ms
        net_ms.set_train()
        def ms_forward(inputs, target):
            out = net_ms(inputs)
            loss = loss_ms_fn(out, target)
            return loss
        
        ms_grad_fn = value_and_grad(ms_forward, self.net_ms.trainable_params())
        loss_ms, grads = ms_grad_fn(mindspore.Tensor(self.inputs), mindspore.Tensor(self.target))
        grads, grad_norm_ms = clip_grad_norm(grads, 1.0)

        assert np.allclose(loss_ms.asnumpy(), loss_pt.detach().numpy(), 1e-4, 1e-4)

        parameters = [p for p in self.net_pt.parameters() if p.grad is not None]
        for grad_ms, param_pt in zip(grads, parameters):
            assert np.allclose(grad_ms.asnumpy(), param_pt.grad.detach().numpy(), 1e-4, 1e-4)