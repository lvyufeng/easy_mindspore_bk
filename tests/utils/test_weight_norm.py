import unittest
import mindspore
import numpy as np
from mindspore import Tensor, context
from easy_mindspore.utils import WeightNorm
import mindspore.nn as nn
import mindspore.common.dtype as mstype

class TestWeightNorm(unittest.TestCase):
    def test_weight_norm_pynative(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        m = WeightNorm(nn.Dense(20, 40))
        assert m.param_g.shape == (40, 1)
        assert m.param_v.shape == (40, 20)
        inputs = Tensor(np.random.randn(10, 20), mstype.float32)
        outputs = m(inputs)
        assert outputs.shape == (10, 40)

    def test_weight_norm_graph(self):
        context.set_context(mode=context.GRAPH_MODE)
        m = WeightNorm(nn.Dense(20, 40))
        assert m.param_g.shape == (40, 1)
        assert m.param_v.shape == (40, 20)
        inputs = Tensor(np.random.randn(10, 20), mstype.float32)
        outputs = m(inputs)
        assert outputs.shape == (10, 40)

    def test_remove_weight_norm_pynative(self):
        context.set_context(mode=context.PYNATIVE_MODE)
        m = WeightNorm(nn.Dense(20, 40))
        assert m.use_weight_norm == True
        assert m.param_g.shape == (40, 1)
        assert m.param_v.shape == (40, 20)
        inputs = Tensor(np.random.randn(10, 20), mstype.float32)
        outputs = m(inputs)
        assert outputs.shape == (10, 40)
        m.remove_weight_norm()
        assert m.use_weight_norm == False

    def test_remove_weight_norm_graph(GRAPH_MODE):
        m = WeightNorm(nn.Dense(20, 40))
        assert m.use_weight_norm == True
        assert m.param_g.shape == (40, 1)
        assert m.param_v.shape == (40, 20)
        inputs = Tensor(np.random.randn(10, 20), mstype.float32)
        outputs = m(inputs)
        assert outputs.shape == (10, 40)
        m.remove_weight_norm()
        assert m.use_weight_norm == False