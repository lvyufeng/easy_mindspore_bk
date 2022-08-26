from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from .pointwise import log, exp, sub
from .comparison import ne, sort
from .others import cast
# argmax
def argmax(input, dim=0, keepdim=False):
    return max(input, dim, keepdim)[1]
# argmin
def argmin(input, dim=0, keepdim=False):
    return min(input, dim, keepdim)[1]
# amax
def amax(input, dim=None, keepdim=False):
    if dim is None:
        dim = ()
    _amax = _get_cache_prim(ops.ReduceMax)(keepdim)
    return _amax(input, dim)
# amin
def amin(input, dim=None, keepdim=False):
    if dim is None:
        dim = ()
    _amin = _get_cache_prim(ops.ReduceMin)(keepdim)
    return _amin(input, dim)
# aminmax
def aminmax(input, dim=None, keepdim=False):
    return amin(input, dim, keepdim), amax(input, dim, keepdim)
# all
def all(input, dim=None, keepdim=False):
    if dim is None:
        dim = ()
    _all = _get_cache_prim(ops.ReduceAll)(keepdim)
    return _all(input, dim)
# any
def any(input, dim=None, keepdim=False):
    if dim is None:
        dim = ()
    _any = _get_cache_prim(ops.ReduceAll)(keepdim)
    return _any(input, dim)
# max
def max(input, dim=0, keepdim=False):
    _max = _get_cache_prim(ops.ArgMaxWithValue)(dim, keepdim)
    arg, val = _max(input)
    return val, arg
# min
def min(input, dim=0, keepdim=False):
    _min = _get_cache_prim(ops.ArgMinWithValue)(dim, keepdim)
    arg, val = _min(input)
    return val, arg
# dist
# logsumexp
def logsumexp(input, dim, keepdim=False):
    _max = amax(input)
    _exp = exp(sub(input, _max))
    _sumexp = sum(_exp, dim, keepdim)
    _logsumexp = log(_sumexp)
    return _logsumexp + _max
# mean
def mean(input, dim=None, keepdim=False, dtype=None):
    if dim is None:
        dim = ()
    _mean = _get_cache_prim(ops.ReduceMean)(keepdim)
    output = _mean(input, dim)
    if dtype is not None:
        return cast(output, dtype)
    return output
# nanmean
# median
def median(input, dim=-1, keepdim=False):
    if dim is None:
        dim = ()
    _median = _get_cache_prim(ops.Median)(False, dim, keepdim)
    return _median(input)
# nanmedian
# mode
# nansum
# prod
def prod(input, dim=None, keepdim=False, dtype=None):
    if dim is None:
        dim = ()
    _prod = _get_cache_prim(ops.ReduceProd)(keepdim)
    output = _prod(input, dim)
    if dtype is not None:
        return cast(output, dtype)
    return output
# quantile
# nanquantile
# std
def std(input, dim=None, unbiased=False, keepdim=False):
    if dim is None:
        dim = ()
    _std = _get_cache_prim(ops.operations.math_ops.ReduceStd)(dim, unbiased, keepdim)
    return _std(input)
# std_mean
# sum
def sum(input, dim=None, keepdim=False, dtype=None):
    if dim is None:
        dim = ()
    _sum = _get_cache_prim(ops.ReduceMean)(keepdim)
    output = _sum(input, dim)
    if dtype is not None:
        return cast(output, dtype)
    return output
# unique
def unique(input, sorted=False, return_inverse=False):
    _unique = _get_cache_prim(ops.Unique)()
    output, inverse_indices = _unique(input)

    if return_inverse:
        if sorted:
            output, indices = sort(output)
            inverse_indices = inverse_indices[indices]
        return output, inverse_indices    
    return output

# unique_consecutive
def unique_consecutive(input, return_inverse=False, return_counts=False, dim=None):
    _unique_consecutive = _get_cache_prim(ops.operations.array_ops.UniqueConsecutive)(return_inverse, return_counts, dim)
    return _unique_consecutive(input)

# var
ops.std
# var_mean
# count_nonzero
def count_nonzero(input, dim=None):
    input_ne = ne(input, 0)
    return sum(input_ne, dim)