from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim

def cast(input, dtype):
    _cast = _get_cache_prim(ops.Cast)()
    return _cast(input, dtype)