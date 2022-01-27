import mindspore.ops as ops
import mindspore.numpy as mnp

def kl_div(input, target, reduction='none', log_target=False):
    if log_target:
        kl_div = ops.exp(target) * (target - input)
    else:
        kl_div = target * (ops.log(target) - input)
    if reduction == 'sum':
        return kl_div.sum()
    if reduction == 'mean':
        return kl_div.mean()
    return kl_div

def cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    pass

def nll_loss(input, target, weight=None, ignore_index=None, reduction='mean'):
    if target.ndim == input.ndim - 1:
        target = target.expand_dims(-1)
    nll_loss = -ops.gather_d(input, -1, target)
    if weight is not None:
        loss_weights = ops.gather(weight, target, 0)
        nll_loss = nll_loss * loss_weights
    else:
        loss_weights = ops.ones_like(nll_loss)
    if ignore_index is not None:
        non_pad_mask = ops.equal(target, ignore_index)
        nll_loss = nll_loss.masked_fill(non_pad_mask, 0.)
        loss_weights = loss_weights.masked_fill(non_pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
    if reduction == 'sum':
        return nll_loss.sum()
    if reduction == 'mean':
        return nll_loss.sum() / loss_weights.sum()
    return nll_loss
