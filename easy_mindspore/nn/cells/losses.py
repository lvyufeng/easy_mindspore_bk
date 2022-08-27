import mindspore.nn as nn
from typing import Optional
from mindspore import Tensor
from ..functional import *

class KLDivLoss(nn.Cell):
    reduction_list = ['sum', 'mean', 'none']
    def __init__(self, reduction:str='mean', log_target:bool=False):
        super().__init__()
        if reduction not in self.reduction_list:
            raise ValueError(f'Unsupported reduction {reduction}')
        self.reduction = reduction
        self.log_target = log_target

    def construct(self, input, target):
        return kl_div(input, target, self.reduction, self.log_target)

class NLLLoss(nn.Cell):
    reduction_list = ['sum', 'mean', 'none']
    def __init__(self, weight: Optional[Tensor]=None, ignore_index:int=-100, reduction:str='mean'):        
        super().__init__()
        if reduction not in self.reduction_list:
            raise ValueError(f'Unsupported reduction {reduction}')
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def construct(self, input, target):
        return nll_loss(input, target, self.weight, self.ignore_index, self.reduction)

class BCEWithLogitsLoss(nn.Cell):
    reduction_list = ['sum', 'mean', 'none']
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean',
                 pos_weight: Optional[Tensor] = None):
        super().__init__()
        if reduction not in self.reduction_list:
            raise ValueError(f'Unsupported reduction {reduction}')
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction

    def construct(self, input, target):
        return binary_cross_entropy_with_logits(input, target, self.weight, self.reduction, self.pos_weight)

class CrossEntropy(nn.Cell):
    reduction_list = ['sum', 'mean', 'none']
    def __init__(self, weight: Optional[Tensor]=None, ignore_index:int=-100, reduction:str='mean', label_smoothing:float=0.0):        
        super().__init__()
        if label_smoothing > 1.0 or label_smoothing < 0.0:
            raise ValueError(f'label_smoothing value must in range [0.0, 1.0], '
                             f'but get {label_smoothing}')
        
        if reduction not in self.reduction_list:
            raise ValueError(f'Unsupported reduction {reduction}')
        
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def construct(self, input, target):
        return cross_entropy(input, target, self.weight, self.ignore_index, self.reduction, self.label_smoothing)

class RDropLoss(nn.Cell):
    reduction_list = ['sum', 'mean', 'none']

    def __init__(self, alpha:float=0.0, reduction:str='none'):
        super().__init__()
        if reduction not in self.reduction_list:
            raise ValueError(f'Unsupported reduction {reduction}')
        self.reduction = reduction
        self.alpha = alpha

    def construct(self, input, input2, target, pad_mask=None):
        ce_loss = 0.5 * (cross_entropy(input, target) + cross_entropy(input2, target))
        kl_loss = self.compute_kl_loss(input, input2, pad_mask)
        return ce_loss + self.alpha * kl_loss
            
    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = kl_div(log_softmax(p, axis=-1), softmax(q, axis=-1), reduction=self.reduction)
        q_loss = kl_div(log_softmax(q, axis=-1), softmax(p, axis=-1), reduction=self.reduction)
        
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill(pad_mask, 0.)
            q_loss.masked_fill(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss