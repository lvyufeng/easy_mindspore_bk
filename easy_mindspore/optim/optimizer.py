from typing import Any
from collections import OrderedDict

import mindspore
from mindspore import log as logger

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()


class Optimizer:
    def __init__(self, params, defaults):
        self.param_groups = []
        self.defaults = defaults

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    @property    
    def parameters(self):
        flatten_params = []
        for param_group in self.param_groups:
            flatten_params.extend([param for param in param_group[0]])

        return flatten_params
    
    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.
        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.
        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"
        assert 'params' in param_group

        params = param_group['params']
        if isinstance(params, mindspore.Parameter):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, mindspore.Parameter):
                raise TypeError("optimizer can only optimize Parameter, "
                                "but one of the params is " + str(type(param)))

        new_param_group = OrderedDict({'params': param_group['params']})
        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            elif name in param_group:
                new_param_group.setdefault(name, param_group[name])
            else:
                new_param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            logger.warning("optimizer contains a parameter group with duplicate parameters")

        param_set = set()
        for group in self.param_groups:
            # group[0] is parameter list
            param_set.update(set(group[0]))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append([v for _, v in new_param_group.items()])
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError
