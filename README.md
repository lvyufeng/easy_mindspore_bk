# Easy MindSpore: a MindSpore warapper for easy usage.

## About EasyMS

## How to use

### Install

```bash
python setup.py install
```

### A functional usage instead of Cell

```python
def train_one_epoch(net, loss_fn, optimizer, dataset, every_print=1, epoch_num=0):
    """train network in one epoch"""
    @ms_function
    def train_step(x, y):
        logits = net(x)
        loss = loss_fn(logits, y)
        return loss, logits, x

    grad_fn = value_and_grad(train_step, net.trainable_params(), has_aux=True)
    steps = 0
    for x, y in dataset.create_tuple_iterator():
        steps += 1
        (loss, _), grads = grad_fn(x, y)
        optimizer(grads)
        if steps % every_print == 0:
            print(f"epoch: {epoch_num}, loss: {loss.asnumpy()}")
```