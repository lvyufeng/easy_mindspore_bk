from mindspore.ops import constexpr

@constexpr
def raise_value_error(info):
    raise ValueError(info)

@constexpr
def raise_runtime_error(info):
    raise RuntimeError(info)