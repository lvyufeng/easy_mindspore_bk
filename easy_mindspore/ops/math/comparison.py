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
# not_equal
# sort
# topk
# msort