import mindspore.numpy as mnp

def kl_div(input, target, reduction='none', log_target=False):
    if log_target:
        kl_div = mnp.exp(target) * (target - input)
    else:
        kl_div = target * (mnp.log(target) - input)
    if reduction == 'sum':
        return kl_div.sum()
    if reduction == 'mean':
        return kl_div.mean()
    return kl_div

def cross_entropy():
    pass