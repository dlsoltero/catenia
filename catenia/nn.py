from __future__ import annotations
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator
from typing import Iterable, Any

import numpy as np

from catenia import Tensor


_default_dtype = np.float32


class Init:

    @staticmethod
    def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
        """Computes fan_in and fan_out for a tensor."""
        dimensions = len(tensor.shape)
        if dimensions < 2:
            raise ValueError("Fan in/out can't be computed for tensor with < 2 dims")

        if dimensions == 2:
            # Linear: (nin, nout)
            fan_in = tensor.shape[0]
            fan_out = tensor.shape[1]
        else:
            # Conv2d: (out_channels, in_channels, kh, kw)
            num_input = tensor.shape[1]
            num_output = tensor.shape[0]
            receptive_field_size = np.prod(tensor.shape[2:])

            fan_in = num_input * receptive_field_size
            fan_out = num_output * receptive_field_size

        return fan_in, fan_out

    @staticmethod
    def _calculate_gain(nonlinearity: str) -> float:
        """Returns the recommended gain value for the given nonlinearity."""
        if nonlinearity == 'relu':
            return np.sqrt(2.0)
        return 1.0

    @staticmethod
    def xavier_normal(tensor: Tensor):
        """
        Fills the input Tensor with values according to the Xavier normal method,
        using a normal distribution with mean 0 and std = gain * sqrt(2 / (fan_in + fan_out)).
        
        This is ideal for layers with no activation or symmetric activations like tanh.
        """
        fan_in, fan_out = Init._calculate_fan_in_and_fan_out(tensor)

        # Xavier standard deviation formula: sqrt(2 / (fan_in + fan_out))
        std = np.sqrt(2.0 / (fan_in + fan_out))

        # In-place update of the tensor's data
        new_data = np.random.normal(0, std, size=tensor.shape)
        tensor.data = new_data.astype(tensor.data.dtype)

        return tensor

    @staticmethod
    def kaiming_uniform(tensor: Tensor, nonlinearity: str = 'relu'):
        """
        Fills the input Tensor with values according to the Kaiming uniform method.
        Good for sigmoid and ReLU, bad for networks with no activation.
        """
        fan_in, _ = Init._calculate_fan_in_and_fan_out(tensor)

        # Calculate standard deviation for Kaiming
        gain = Init._calculate_gain(nonlinearity)
        std = gain / np.sqrt(fan_in)

        # Calculate boundary for uniform distribution: sqrt(3) * std
        bound = np.sqrt(3.0) * std

        # In-place update of the tensor's data
        new_data = np.random.uniform(-bound, bound, size=tensor.shape)
        tensor.data = new_data.astype(tensor.data.dtype)

        return tensor

    @staticmethod
    def bias_uniform(bias: Tensor, weight: Tensor):
        """
        Fills the bias tensor with values from U(-bound, bound),
        where bound = 1 / sqrt(fan_in).
        """
        fan_in, _ = Init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0

        target_dtype = bias.data.dtype
        new_data = np.random.uniform(-bound, bound, size=bias.shape)
        bias.data = new_data.astype(target_dtype)
        return bias


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

    def state_dict(self) -> OrderedDict:
        """Returns a dictionary containing a whole state of the module."""
        state = OrderedDict()
        for name, param in self.named_parameters():
            state[name] = param
        return state

    def load_state_dict(self, state_dict: dict, quiet=False):
        """Copies parameters from state_dict into this module."""
        own_state = self.state_dict()
        loaded_keys = set()
        for name, param in state_dict.items():
            if name in own_state:
                # Update data in-place
                own_state[name].data = param.data
                loaded_keys.add(name)
            else:
                raise KeyError(f"Unexpected key {name} in state_dict")

        all_keys = set(own_state.keys())
        missing_keys = all_keys - loaded_keys

        if missing_keys and not quiet:
            print(f"WARNING: The following keys from the model were NOT updated: {list(missing_keys)}")


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


class Sequential(Module):
    """
    A sequential container. Modules will be added to it in the order 
    they are passed in the constructor.
    """
    def __init__(self, *args: Module):
        super().__init__()
        # We use ModuleList to track the parameters of every layer
        self.layers = ModuleList(args)

    def forward(self, x: Tensor) -> Tensor:
        # Pass the input through each layer in the sequence
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)


class Linear(Module):
    """Fully connected layer"""

    def __init__(self, nin: int, nout: int, dtype=None):
        super().__init__()
        target_dtype = dtype if dtype is not None else _default_dtype

        self.weight = Parameter(np.empty((nin, nout), dtype=target_dtype))
        self.bias = Parameter(np.empty(nout, dtype=target_dtype))

        # Initialize weight
        Init.kaiming_uniform(self.weight, nonlinearity='relu')
        Init.bias_uniform(self.bias, self.weight)

    def forward(self, x):
        return x @ self.weight + self.bias
        # return out[0] if len(out) == 1 else out


class Conv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size: int | tuple, stride=1, padding=0, dtype=None):
        super().__init__()
        self.stride = stride
        self.padding = padding
        target_dtype = dtype if dtype is not None else _default_dtype

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        # Shape: (out_channels, in_channels, kh, kw)
        weight_shape = (out_channels, in_channels, *kernel_size)

        self.weight = Parameter(np.empty(weight_shape, dtype=target_dtype))
        self.bias = Parameter(np.empty(out_channels, dtype=target_dtype))

        Init.kaiming_uniform(self.weight, nonlinearity='relu')
        Init.bias_uniform(self.bias, self.weight)

    def forward(self, x: Tensor) -> Tensor:
        return x.conv2d(self.weight, self.bias, self.stride, self.padding)


class Flatten(Module):

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)


class ReLU(Module):

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


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


class CrossEntropyLoss(Module):

    def __init__(self, reduction: str | None = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            logits: Predicted raw scores (Batch, Classes)
            target: Ground truth indices (Batch,) or One-hot (Batch, Classes)
        """
        # Log-Sum-Exp trick for stability: log(sum(exp(x)))
        # Subtract max for numerical stability: exp(x - max)
        max_logits = logits.max(axis=1, keepdims=True)
        stable_logits = logits - max_logits
        log_sum_exp = stable_logits.exp().sum(axis=1, keepdims=True).log()
        
        # Log Softmax
        log_probs = stable_logits - log_sum_exp

        # Extract the log-probability of the true class (NLL)
        # If target is one-hot, we can just multiply and sum
        if logits.shape == target.shape:
            # NLL Loss (Assumes target is one-hot or same shape as logits)
            loss = -(target * log_probs).sum(axis=1)
        else:
            # If target is indices, you'd ideally use advanced indexing
            # Assuming your Tensor handles basic selection or target is converted
            # For simplicity in this snippet, we treat target as one-hot
            raise NotImplementedError("Implement indexing for integer targets or pass one-hot.")

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
