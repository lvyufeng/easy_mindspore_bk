import unittest
from easy_mindspore.core.api import value_and_grad
import mindspore.numpy as mnp
import numpy as np
import mindspore

class TestValueAndGrad(unittest.TestCase):
    def test_value_and_grad(self):
        def forward(x):
            y = 2 * mnp.dot(x, x)
            return y
        x = mnp.arange(4.0, dtype=mindspore.float32)
        grad_fn = value_and_grad(forward)
        values, grads = grad_fn(x)
        print(grads[0].asnumpy())
        assert np.array_equal(values.asnumpy(), np.array(28.0))
        assert np.array_equal(grads[0].asnumpy(), np.array([0.0, 4.0, 8.0, 12.0]))

    def test_value_and_grad_has_aux(self):
        def forward(x):
            y = 2 * mnp.dot(x, x)
            return y, x
        x = mnp.arange(4.0, dtype=mindspore.float32)
        grad_fn = value_and_grad(forward, has_aux=True)
        values, grads = grad_fn(x)
        print(grads[0].asnumpy())
        assert np.array_equal(values[0].asnumpy(), np.array(28.0))
        assert np.array_equal(grads[0].asnumpy(), np.array([0.0, 4.0, 8.0, 12.0]))