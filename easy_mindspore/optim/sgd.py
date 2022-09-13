import mindspore.ops as ops
from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)
    
    def __call__(self, grads):
        return True

def sgd():
    pass

