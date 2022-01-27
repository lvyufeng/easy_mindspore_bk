import mindspore.nn as nn
import mindspore.ops as ops
from .functional import kl_div

class KLDivLoss(nn.Cell):
    reduction_list = ['sum', 'mean', 'none']
    def __init__(self, reduction='mean', log_target=False):
        super().__init__()
        if reduction not in self.reduction_list:
            raise ValueError(f'Unsupported reduction {reduction}')
        self.reduction = reduction
        self.log_target = log_target

    def construct(self, input, target):
        return kl_div(input, target, self.reduction, self.log_target)

class RDropLoss(nn.Cell):
    def __init__(self, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)

    def construct(self, logits, logits2, label):
        return None