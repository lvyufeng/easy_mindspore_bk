import mindspore.ops as ops

# Non-linear activation functions
def threshold(input, threshold):
    return ops.maximum(input, threshold)

def relu(input):
    return ops.ReLU()(input)

def hardtanh(input, min_val=- 1.0, max_val=1.0):
    return ops.maximum(ops.minimum(input, max_val), min_val)

def hardswish(input):
    return input * ops.ReLU6(input + 3) / 6

def relu6(input):
    return ops.ReLU6()(input)

def elu(input, alpha=1.0):
    return ops.Elu(alpha)(input)

def selu(input):
    return 1.0507009873554804934193349852946 * \
        (ops.maximum(0, input) + ops.minimum(0, 1.6732632423543772848170429916717 * (ops.exp(input) - 1)))

def celu(input, alpha=1.0):
    return ops.maximum(0, input) + ops.minimum(0, alpha * (ops.exp(input / alpha) - 1))

def leaky_relu(input, negative_slope=0.01):
    return ops.maximum(0, input) + negative_slope * ops.minimum(0, input)

def prelu(input, weight):
    return ops.maximum(0, input) + weight * ops.minimum(0, input)

def rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False):
    if training:
        alpha = ops.uniform(input.shape, lower, upper)
    else:
        alpha = (upper + lower) / 2
    return relu(input) - relu(-input) * alpha

def glu(input, dim=- 1):
    a, b = ops.Split(dim, 2)(input)
    return a * ops.Sigmoid()(b)

def gelu(input):
    return ops.GeLU()(input)

def logsigmoid(input):
    return ops.log(1 / (1 + ops.exp(-input)))

def hardshrink(input, lambd=0.5):
    great_lambd = ops.Greater()(input, lambd)
    less_neg_lambd = ops.Less()(input, lambd)
    cond = ops.logical_or(great_lambd, less_neg_lambd)
    return ops.Select()(cond, input, ops.scalar_to_tensor(0.0))

def tanhshrink(input):
    return input - tanh(input)

def softsign():
    pass

def softplus():
    pass

def softmin():
    pass

def softmax(input, axis=-1):
    return ops.Softmax(axis)(input)

def softshrink():
    pass

def gumbel_softmax():
    pass

def log_softmax(input, axis=-1):
    return ops.LogSoftmax(axis)(input)

def tanh():
    pass

def sigmoid():
    pass

def hardsigmoid():
    pass

def silu():
    pass

def mish():
    pass

def batch_norm():
    pass

def group_norm():
    pass

def layer_norm():
    pass

def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1.):
    # type: (Tensor, int, float, float, float) -> Tensor
    r"""Applies local response normalization over an input signal composed of
    several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    See :class:`~torch.nn.LocalResponseNorm` for details.
    """
    dim = input.dim() # 重点！
    if dim < 3:
        raise ValueError('Expected 3D or higher dimensionality \
                         input (got {} dimensions)'.format(dim))
    div = input.mul(input).unsqueeze(1) # 重点！
    if dim == 3:
        div = pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else: # 重点！
        sizes = input.size() # 重点！
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1) # 重点！
        div = pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2)) # 重点！
        div = avg_pool3d(div, (size, 1, 1), stride=1).squeeze(1) # 重点！
        div = div.view(sizes) # 重点！
    div = div.mul(alpha).add(k).pow(beta) # 重点！
    return input / div # 重点！

def normalize():
    pass

# Loss functions
def kl_div(input, target, reduction='none', log_target=False):
    if log_target:
        kl_div = ops.exp(target) * (target - input)
    else:
        output = target * (ops.log(target) - input)
        zeros = ops.zeros_like(input)
        kl_div = ops.select(target > 0, output, zeros)
    if reduction == 'sum':
        return kl_div.sum()
    if reduction == 'mean':
        return kl_div.mean()
    return kl_div

def cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    return nll_loss(log_softmax(input, 1), target, weight, ignore_index, reduction, label_smoothing)

def nll_loss(input, target, weight=None, ignore_index=None, reduction='mean', label_smoothing=0.0):
    ndim = input.ndim
    if ndim == 2:
        ret = _nll_loss(input, target, -1, weight, ignore_index, reduction, label_smoothing)
    elif input.ndim == 4:
        ret = _nll_loss(input, target, 1, weight, ignore_index, reduction, label_smoothing)
    else:
        # ndim == 3 or ndim > 4
        n = input.shape[0]
        c = input.shape[1]
        out_size = (n,) + input.shape[2:]
        input = input.view(n, c, 1, -1)
        target = target.view(n, 1, -1)
        if reduction != 'none':
            ret = _nll_loss(input, target, 1, weight, ignore_index, reduction, label_smoothing)
        else:
            ret = _nll_loss(input, target, 1, weight, ignore_index, label_smoothing=label_smoothing)
            ret = ret.view(out_size)
    return ret

def _nll_loss(input, target, target_dim=-1, weight=None, ignore_index=None, reduction='none', label_smoothing=0.0):
    if target.ndim == input.ndim - 1:
        target = target.expand_dims(target_dim)
    nll_loss = -ops.gather_d(input, target_dim, target)
    smooth_loss = -input.sum(axis=target_dim, keepdims=True)
    if weight is not None:
        loss_weights = ops.gather(weight, target, 0)
        nll_loss = nll_loss * loss_weights
    else:
        loss_weights = ops.ones_like(nll_loss)
    if ignore_index is not None:
        non_pad_mask = ops.equal(target, ignore_index)
        nll_loss = nll_loss.masked_fill(non_pad_mask, 0.)
        loss_weights = loss_weights.masked_fill(non_pad_mask, 0.)
        smooth_loss = smooth_loss.masked_fill(non_pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(target_dim)
        smooth_loss = smooth_loss.squeeze(target_dim)

    if reduction == 'sum':
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == 'mean':
        nll_loss = nll_loss.sum() / loss_weights.sum()
        smooth_loss = smooth_loss.mean()
    
    eps_i = label_smoothing / input.shape[target_dim]
    loss = (1. - label_smoothing) * nll_loss + eps_i * smooth_loss

    return loss

def binary_cross_entropy(input, target, weight=None, reduction='mean'):
    pass

def binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean', pos_weight=None):
    max_val = ops.maximum(-input, 0)

    if pos_weight is not None:
        log_weight = ((pos_weight - 1) * target) + 1
        loss = (1 - target) * input
        loss_1 = ops.log(ops.exp(-max_val) + ops.exp(-input - max_val)) + max_val
        loss += log_weight * loss_1
    else:
        loss = (1 - target) * input
        loss += max_val
        loss += ops.log(ops.exp(-max_val) + ops.exp(-input - max_val))
 
    if weight is not None:
        output = loss * weight
    else:
        output = loss

    if reduction == "mean":
        return ops.reduce_mean(output)
    elif reduction == "sum":
        return ops.reduce_sum(output)
    else:
        return output
