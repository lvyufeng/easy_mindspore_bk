import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.common.seed import get_seed
# bernoulli
# multinomial
# normal
# poisson
# rand
# rand_like
# randint
# randint_like
# randn
def randn(*size, dtype=None):
    seed = get_seed()
    if seed is not None:
        _std_normal = _get_cache_prim(ops.StandardNormal)(seed=seed)
    else:
        _std_normal = _get_cache_prim(ops.StandardNormal)()
    if dtype is not None:
        return _std_normal(size).astype(dtype)
    return _std_normal(size)
# randn_like
# randperm