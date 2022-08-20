from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from .comparison import maximum, minimum
from ..utils import raise_value_error

# abs
def abs(input):
    _abs = _get_cache_prim(ops.Abs)()
    return _abs(input)
# absolute
absolute = abs

# acos
def acos(input):
    _acos = _get_cache_prim(ops.ACos)()
    return _acos(input)
# arccos
arccos = acos
# acosh
def acosh(input):
    _acosh = _get_cache_prim(ops.Acosh)()
    return _acosh(input)
# arccosh
arccosh = acosh
# add
def add(input, other, alpha=1):
    return input + alpha * other
# addcdiv
def addcdiv(input, tensor1, tensor2, value=1):
    _addcdiv = _get_cache_prim(ops.Addcdiv)()
    return _addcdiv(input, tensor1, tensor2, value)
# addcmul
def addcmul(input, tensor1, tensor2, value=1):
    _addcmul = _get_cache_prim(ops.Addcdiv)()
    return _addcmul(input, tensor1, tensor2, value)
# angle
# asin
def asin(input):
    _asin = _get_cache_prim(ops.Asin)()
    return _asin(input)
# arcsin
arcsin = asin
# asinh
def asinh(input):
    _asinh = _get_cache_prim(ops.Asinh)()
    return _asinh(input)
# arcsinh
arcsinh = asinh
# atan
def atan(input):
    _atan = _get_cache_prim(ops.Atan)()
    return _atan(input)
# arctan
arctan = atan
# atanh
def atanh(input):
    _atanh = _get_cache_prim(ops.Atanh)()
    return _atanh(input)
# arctanh
arctanh = atanh
# atan2
def atan2(input):
    _atan2 = _get_cache_prim(ops.Atan2)()
    return _atan2(input)
# arctan2
arctan2 = atan2
# bitwise_not
# bitwise_and
def bitwise_and(input, other):
    _bitwise_and = _get_cache_prim(ops.BitwiseAnd)()
    return _bitwise_and(input, other)
# bitwise_or
def bitwise_or(input, other):
    _bitwise_or = _get_cache_prim(ops.BitwiseOr)()
    return _bitwise_or(input, other)
# bitwise_xor
def bitwise_xor(input, other):
    _bitwise_xor = _get_cache_prim(ops.BitwiseXor)()
    return _bitwise_xor(input, other)
# bitwise_left_shift
def bitwise_left_shift(input, other):
    _bitwise_left_shift = _get_cache_prim(ops.operations.array_ops.LeftShift)()
    return _bitwise_left_shift(input, other)
# bitwise_right_shift
def bitwise_right_shift(input, other):
    _bitwise_right_shift = _get_cache_prim(ops.operations.array_ops.RightShift)()
    return _bitwise_right_shift(input, other)
# ceil
# clamp
def clamp(input, min=None, max=None):
    if min is None and max is None:
        return input
    if min is not None and max is not None:
        return minimum((maximum(input, min)), max)
    if min is not None:
        return maximum(input, min)
    if max is not None:
        return minimum(input, max)

# clip
clip = clamp
# conj_physical
# copysign
# cos
# cosh
# deg2rad
# div
def div(input, other, rounding_mode=None):
    if rounding_mode is None:
        _div = _get_cache_prim(ops.RealDiv)()
    elif rounding_mode == 'floor':
        _div = _get_cache_prim(ops.FloorDiv)()
    elif rounding_mode == 'trunc':
        _div = _get_cache_prim(ops.TruncateDiv)()
    else:
        raise_value_error(f'Do not support "rounding_mode": {rounding_mode}.')

    return _div(input, other)
# divide
def divide(input, other, rounding_mode=None):
    return div(input, other, rounding_mode)
# digamma
# erf
# erfc
# erfinv
# exp
# exp2
# expm1
# fake_quantize_per_channel_affine
# fake_quantize_per_tensor_affine
# fix
# float_power
# floor
# floor_divide
# fmod
# frac
# frexp
# gradient
# imag
# ldexp
# lerp
# lgamma
# log
def log(input):
    _log = _get_cache_prim(ops.Log)()
    return _log(input)
# log10
# log1p
# log2
# logaddexp
# logaddexp2
# logical_and
# logical_not
# logical_or
# logical_xor
# hypot
# mul
# multiply
# nan_to_num
# neg
# negative
# nextafter
# positive
# pow
# quantized_batch_norm
# quantized_max_pool1d
# quantized_max_pool2d
# rad2deg
# real
# reciprocal
# remainder
# round
# rsqrt
# sign
# sgn
# signbit
# sin
# sinh
# sqrt
# square
# sub
# subtract
# tan
# tanh
# true_divide
def true_divide(input, other):
    return div(input, other, None)
# trunc
def trunc(input):
    _trunc = _get_cache_prim(ops.Trunc)()
    return _trunc(input)
