from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from .math.pointwise import clamp, log
# logit
def logit(input, eps=None):
    if eps is not None:
        input = clamp(input, eps, 1-eps)
    return log(input) - log(1-input)        
# i0
# igamma
# igammac
def multigammaln(input, p):
    _mvlgamma = _get_cache_prim(ops.Mvlgamma)(p)
    return _mvlgamma(input)
# polygamma

# expit
def expit(input):
    _expit = _get_cache_prim(ops.Sigmoid)()
    return _expit(input)

# sinc
def sinc(input):
    _sinc = _get_cache_prim(ops.Sinc)
# xlogy
def xlogy(input, other):
    _xlogy = _get_cache_prim(ops.Xlogy)()
    return xlogy(input, other)
