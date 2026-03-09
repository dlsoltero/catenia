from __future__ import annotations
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator
from typing import Iterable, Any

import numpy as np

from catenia import Tensor


class Init:

    @staticmethod
    def unactivated(nin: int, nout: int, dtype):
        """
        Avoid exploding/vanishing gradient by keeping mean 0 and std 1 in a
        Linear layer with no activation

        Args:
            nin: Fan-in or incoming network connections
            nout: Fan-out or outgoing network connections

        Return:
            Normalize weights, bias
        """
        weights = np.random.normal(0, 1, size=(nin, nout)) * np.sqrt(1/nin)
        biases = np.zeros(nout)
        return Tensor(weights, dtype=dtype), Tensor(biases, dtype=dtype)

    @staticmethod
    def kaiming_uniform(nin: int, nout: int, dtype, nonlinearity='relu'):
        """
        Kaiming/He initialization for ReLU-based networks.

        Good for sigmoid and relu, bad for networks with no activation.

        Args:
            nin: Fan-in or incoming network connections
            nout: Fan-out or outgoing network connections
        
        Return:
            Normalize weights, bias
        """
        # Calculation of 'gain' for ReLU is sqrt(2)
        gain = np.sqrt(2.0) if nonlinearity == 'relu' else 1.0
        
        # Standard deviation for Kaiming is gain / sqrt(fan_in)
        # For uniform distribution, the limit is sqrt(3) * std
        std = gain / np.sqrt(nin)
        limit = np.sqrt(3.0) * std  # This simplifies to np.sqrt(6.0 / nin)
        
        w = np.random.uniform(low=-limit, high=limit, size=(nin, nout))
        b = np.zeros(nout)
        
        return Tensor(w, dtype=dtype), Tensor(b, dtype=dtype)


class Parameter(Tensor):
    def __init__(self, data, dtype=None) -> None:
        # If data is already a Tensor, use its attributes
        if isinstance(data, Tensor):
            target_dtype = dtype if dtype is not None else data.data.dtype
            super().__init__(
                data=data.data, 
                dtype=target_dtype, 
                _children=data._prev, 
                _op=data._op
            )
        else:
            super().__init__(data=data, dtype=dtype)


class Module(ABC):

    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    @abstractmethod
    def forward(self, *args, **kargs):
        """Subclass must implement the method"""
        pass

    def __call__(self, *args, **kargs):
        return self.forward(*args, **kargs)
    
    def __setattr__(self, name: str, value: Any) -> None:
        params = self.__dict__.get('_parameters')
        mods = self.__dict__.get('_modules')

        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("Base Module not initialized. Did you forget super().__init__()?")
            params[name] = value
        elif isinstance(value, Module):
            if mods is None:
                raise AttributeError("Base Module not initialized. Did you forget super().__init__()?")
            mods[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Tensor | 'Module':
        params = self.__dict__['_parameters']
        if name in params:
            return params[name]

        modules = self.__dict__['_modules']
        if name in modules:
            return modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def named_modules(self, prefix: str = '') -> Iterator[tuple[str, 'Module']]:
        yield prefix, self
        for name, module in self._modules.items():
            submodule_prefix = prefix + ('.' if prefix else '') + name
            yield from module.named_modules(submodule_prefix)

    def modules(self) -> Iterator['Module']:
        for name, module in self.named_modules():
            yield module

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[tuple[str, Parameter]]:
        gen = self._named_members(lambda module: module._parameters.items(), prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    # Recursion may cause an optimizer to update the same parameter multiple times
    # in complex neural networks sharing weights between layers

    # def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    #     for _, param in self.named_parameters(recurse=recurse):
    #         yield param

    def parameters(self):
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class ModuleList(Module):

    def __init__(self, modules: Iterable[Module] = None):
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def __setitem__(self, idx: int, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError(f"ModuleList can only contain Modules, got {type(module)}")
        # Use setattr to trigger the registration logic in the base Module class
        setattr(self, str(idx), module)

    def __getitem__(self, idx: int) -> Module:
        # We store them in _modules with string keys to satisfy the base class
        return self._modules[str(idx)]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def append(self, module: Module) -> 'ModuleList':
        self[len(self)] = module
        return self

    def extend(self, modules: Iterable[Module]) -> 'ModuleList':
        for module in modules:
            self.append(module)
        return self

    def forward(self, *args, **kwargs):
        # ModuleList itself doesn't have a forward pass logic, it's just a container.
        raise NotImplementedError("ModuleList is a container and does not implement forward().")


class Linear(Module):
    """Fully connected layer"""

    def __init__(self, nin, nout, dtype=None):
        super().__init__()
        w, b = Init.kaiming_uniform(nin=nin, nout=nout, dtype=dtype)
        self.w = Parameter(w)
        self.b = Parameter(b)

    def forward(self, x):
        return x @ self.w + self.b
        # return out[0] if len(out) == 1 else out


class MSELoss(Module):

    def __init__(self, reduction: str | None = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        result = (pred - target) ** 2
        if self.reduction is None:
            return result
        elif self.reduction == 'mean':
            return result.mean()
        elif self.reduction == 'sum':
            return result.sum()
        else:
            raise ValueError("reduction valid values are: mean, sum, or None")
