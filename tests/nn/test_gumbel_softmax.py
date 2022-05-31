import unittest
import mindspore
from mindspore import Tensor, context
from easy_mindspore.nn import GumbelSoftmax
from easy_mindspore import randn

class TestGumbelSoftmax(unittest.TestCase):
    def test_gumbel_softmax(self):
        gumbel_softmax = GumbelSoftmax(temperature=0.01)
        logits = randn(4, 6)
        gumbels = gumbel_softmax(logits)
        print(gumbels)
        assert gumbels.shape == (4, 6)
    
    def test_gumbel_softmax_hard(self):
        gumbel_softmax = GumbelSoftmax(temperature=0.2, hard=True)
        logits = randn(20, 32)
        gumbels = gumbel_softmax(logits)
        assert gumbels.shape == (20, 32)

    # def test_gumbel_softmax_sample(self):
    #     gumbel_softmax = GumbelSoftmax(temperature=1000)
    #     logits = Tensor([0.7, 0.2, 0.1], mindspore.float32)
    #     q = {0:0,1:0,2:0}
    #     for _ in range(10000): # 进行一万次采样
    #         t = gumbel_softmax(logits).argmax().asnumpy()
    #         q[int(t)] += 1
    #     softmax = mindspore.ops.Softmax()(logits)
    #     print(q, softmax)

    # def test_gumbel_softmax_sample_hard(self):
    #     gumbel_softmax = GumbelSoftmax(hard=True)
    #     logits = Tensor([0.7, 0.2, 0.1], mindspore.float32)
    #     q = {0:0,1:0,2:0}
    #     for _ in range(10000): # 进行一万次采样
    #         t = gumbel_softmax(logits).argmax().asnumpy()
    #         q[int(t)] += 1
    #     softmax = mindspore.ops.Softmax()(logits)
    #     print(q, softmax)