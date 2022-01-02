from mindspore._c_expression import GradOperation_
from mindspore.common.api import ms_function

def value_and_grad(fn, params=None, has_aux=False):
    if params is None:
        grad_ = GradOperation_('grad', True, False, False, False)
    else:
        grad_ = GradOperation_('grad', False, True, False, False)

    def fn_aux(*args):
        return fn(*args)[0]

    if has_aux:
        fn_ = fn_aux
    else:
        fn_ = fn

    @ms_function
    def value_and_grad_f(*args):
        values = fn(*args)
        if params is None:
            grads = grad_(fn_)(*args)
        else:
            grads = grad_(fn_, params)(*args)
        return values, grads
    return value_and_grad_f

def grad(fn, has_aux=False):
    value_and_grad_f = value_and_grad(fn, has_aux)
    def grad_f(*args):
        _, g = value_and_grad_f(*args)
        return g
    return grad_f