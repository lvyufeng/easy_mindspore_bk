from functools import partial
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from .comparison import maximum, minimum
from ..utils import raise_value_error
from mindspore import Tensor

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
def ceil(input):
    _ceil = _get_cache_prim(ops.Ceil)()
    return _ceil(input)
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
def conj_physical(input):
    _conj = _get_cache_prim(ops.Conj)()
    return _conj(input)
# copysign
# cos
def cos(input):
    _cos = _get_cache_prim(ops.Cos)()
    return _cos(input)
# cosh
def cosh(input):
    _cosh = _get_cache_prim(ops.Cosh)()
    return _cosh
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
divide = div
# true_divide
true_divide = partial(div, rounding_mode=None)
# digamma
# erf
def erf(input):
    _erf = _get_cache_prim(ops.Erf)()
    return _erf(input)
# erfc
def erfc(input):
    _erfc = _get_cache_prim(ops.Erfc)()
    return _erfc(input)
# erfinv
def erfinv(input):
    _erfinv = _get_cache_prim(ops.Erfinv)()
    return _erfinv(input)
# exp
def exp(input):
    _exp = _get_cache_prim(ops.Exp)()
    return _exp(input)
# exp2
def exp2(input):
    return pow(Tensor(2, input.dtype), input)
# expm1
def expm1(input):
    _expm1 = _get_cache_prim(ops.Expm1)()
    return _expm1(input)
# fake_quantize_per_channel_affine
# fake_quantize_per_tensor_affine
# float_power
# floor
def floor(input):
    _floor = _get_cache_prim(ops.Floor)()
    return _floor(input)
# floor_divide
def floor_divide(input, other):
    _floor_divide = _get_cache_prim(ops.FloorDiv)()
    return _floor_divide(input, other)
# fmod
def fmod(input, other):
    # return sub(input, div(input, other, rounding_mode="trunc") * other)
    _fmod = _get_cache_prim(ops.FloorMod)()
    return _fmod(input, other)
# frac
def frac(input):
    frac_op = _get_cache_prim(ops.Mod)()
    return frac_op(input, 1)
# frexp
# gradient
# imag
def imag(input):
    _imag = _get_cache_prim(ops.Imag)()
    return _imag(input)
# ldexp
def ldexp(input, other):
    return mul(input, pow(2.0, other))
# lerp
def lerp(input, end, weight):
    _lerp = _get_cache_prim(ops.Lerp)()
    return _lerp(input, end, weight)
# lgamma
# log
def log(input):
    _log = _get_cache_prim(ops.Log)()
    return _log(input)
# log10
def log10(input):
    _log = _get_cache_prim(ops.Log)()
    return _log(sub(input, Tensor(10, input.dtype)))
# log1p
def log1p(input):
    _log1p = _get_cache_prim(ops.Log1p)()
    return _log1p(input)
# log2
def log10(input):
    _log = _get_cache_prim(ops.Log)()
    return _log(sub(input, Tensor(2, input.dtype)))
# logaddexp
def logaddexp(input, other):
    return log(exp(input), exp(other))
# logaddexp2
def logaddexp2(input, other):
    return log(exp2(input), exp2(other))
# logical_and
def logical_and(input, other):
    _logical_and = _get_cache_prim(ops.LogicalAnd)()
    return _logical_and(input, other)
# logical_not
def logical_not(input):
    _logical_not = _get_cache_prim(ops.LogicalNot)()
    return _logical_not(input)
# logical_or
def logical_or(input, other):
    _logical_or = _get_cache_prim(ops.LogicalOr)()
    return _logical_or(input, other)
# logical_xor
def logical_xor(input, other):
    _logical_xor = _get_cache_prim(ops.LogicalXor)()
    return _logical_xor(input, other)
# hypot
# mul
def mul(input, other):
    _mul = _get_cache_prim(ops.Mul)()
    return _mul(input, other)
# multiply
multiply = mul
# nan_to_num
# neg
def neg(input):
    _neg = _get_cache_prim(ops.Neg)()
    return _neg(input)
# negative
negative = neg
# nextafter
# positive
# pow
def pow(input, exponent):
    _pow = _get_cache_prim(ops.Pow)()
    return _pow(input, exponent)
# quantized_batch_norm
# quantized_max_pool1d
# quantized_max_pool2d
# rad2deg
# real
def real(input):
    _real = _get_cache_prim(ops.Real)()
    return _real(input)
# reciprocal
def reciprocal(input):
    _reciprocal = _get_cache_prim(ops.Reciprocal)()
    return _reciprocal(input)
# remainder
# round
def round(input):
    _round = _get_cache_prim(ops.Round)()
    return _round(input)
# rsqrt
def rsqrt(input):
    _rsqrt = _get_cache_prim(ops.Rsqrt)()
    return _rsqrt(input)
# sign
def sign(input):
    _sign = _get_cache_prim(ops.Sign)()
    return _sign(input)
# sgn
# signbit
# sin
def sin(input):
    _sin = _get_cache_prim(ops.Sin)()
    return _sin(input)
# sinh
def sinh(input):
    _sinh = _get_cache_prim(ops.Sinh)()
    return _sinh(input)
# sqrt
def sqrt(input):
    _sqrt = _get_cache_prim(ops.Sqrt)()
    return _sqrt(input)
# square
def square(input):
    _square = _get_cache_prim(ops.Square)()
    return _square(input)
# sub
def sub(input, other, alpha=1):
    _sub = _get_cache_prim(ops.Sub)()
    return _sub(input, mul(alpha, other))
# subtract
substract = sub
# tan
def tan(input):
    _tan = _get_cache_prim(ops.Tan)()
    return _tan
# tanh
def tanh(input):
    _tanh = _get_cache_prim(ops.Tanh)()
    return _tanh(input)
# trunc
def trunc(input):
    _trunc = _get_cache_prim(ops.Trunc)()
    return _trunc(input)
# fix
fix = trunc

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
mvlgamma = multigammaln
# polygamma

# expit
def sigmoid(input):
    _expit = _get_cache_prim(ops.Sigmoid)()
    return _expit(input)
expit = sigmoid
# sinc
# xlogy
def xlogy(input, other):
    _xlogy = _get_cache_prim(ops.Xlogy)()
    return _xlogy(input, other)