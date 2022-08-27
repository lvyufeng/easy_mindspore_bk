from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
# allclose
# argsort
# eq
# equal
# ge
# greater_equal
# gt
# greater
# isclose
# isfinite
# isin
# isinf
# isposinf
# isneginf
# isnan
# isreal
# kthvalue
# le
# less_equal
# lt
# less
# maximum
def maximum(input, other):
    _maximum = _get_cache_prim(ops.Maximum)()
    return _maximum(input, other)
# minimum
def minimum(input, other):
    _minimum = _get_cache_prim(ops.Minimum)()
    return _minimum(input, other)
# fmax
# fmin
# ne
def ne(input, other):
    _ne = _get_cache_prim(ops.NotEqual)()
    return _ne(input, other)
# not_equal
not_equal = ne
# sort
def sort(input, dim=-1, descending=False):
    _sort = _get_cache_prim(ops.Sort)(dim, descending)
    return _sort(input)
# topk
# msort