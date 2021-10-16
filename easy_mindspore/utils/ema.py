import mindspore.nn as nn
import mindspore.ops as P
import mindspore.common.dtype as mstype
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size, get_rank

class TrainingWrapperEMA(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.sens = sens

        self.network.set_grad()
        self.weights = optimizer.parameters
        self.grad = P.GradOperation(get_by_list=True, sens_param=True)
        self.reduce_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [context.ParallelMode.DATA_PARALLEL, context.ParallelMode.HYBRID_PARALLEL]:
            self.reduce_flag = True
        if self.reduce_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context('device_num')
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

        self.lens = len(self.weights)
        self.weights_keep = self.weights.clone(prefix="new_")
        self.assign_add = P.AssignAdd()
        self.assign = P.Assign()
        self.exp = P.Exp()
        self.global_steps = Parameter(Tensor(1.0, mstype.float32), name='global_steps')

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        if self.reduce_flag:
            grads = self.grad_reducer(grads)
        loss = P.depend(loss, self.optimizer(grads))

        factor = 1 - self.exp(-self.global_steps / 2000)
        for i in range(self.lens):
            self.assign(self.weights_keep[i], factor * self.weights_keep[i])
            self.assign_add(self.weights_keep[i], (1 - factor) * self.weights_keep[i])
        
        self.assign_add(self.global_steps, 1.)
        
        return loss
