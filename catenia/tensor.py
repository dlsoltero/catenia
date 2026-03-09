import numpy as np


def ensure_ndarray(data, dtype) -> np.ndarray:
    """Turn data into a numpy array of the specified dtype."""
    if isinstance(data, Tensor):
        arr = data.data
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        arr = np.array(data, dtype=dtype)

    target_dtype = dtype if dtype is not None else np.float32

    if arr.dtype != target_dtype:
        arr = arr.astype(target_dtype)

    return arr

def ensure_tensor(data, dtype) -> np.ndarray:
    if isinstance(data, Tensor):
        return data
    return Tensor(data, dtype=dtype)

def _unbroadcast(grad: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Unbroadcast gradients to match the shape of the original tensor."""
    if grad.shape == original_shape:
        return grad

    # Pad original_shape with leading 1s to match grad.ndim
    ndim_added = grad.ndim - len(original_shape)
    padded_shape = (1,) * ndim_added + tuple(original_shape)

    # Collect all axes to sum: either added leading axes, or axes where
    # original was 1 (and thus broadcast-expanded)
    sum_axes = tuple(
        i for i, (g, o) in enumerate(zip(grad.shape, padded_shape))
        if o == 1
    )

    if sum_axes:
        grad = grad.sum(axis=sum_axes, keepdims=True)

    # Strip the leading size-1 dims that were added
    if ndim_added > 0:
        grad = grad.reshape(grad.shape[ndim_added:])

    return grad.reshape(original_shape)


class Tensor:

    def __init__(
        self,
        data,
        dtype=None,
        _children: tuple = (),
        _op: str = '' 
    ):
        self.data = ensure_ndarray(data, dtype=dtype)
        self.grad = np.zeros(self.data.shape, dtype=dtype)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def dtype(self) -> str:
        return self.data.dtype

    def __repr__(self):
        data_str = np.array2string(
            self.data,
            precision=4,
            separator=", ",
            suppress_small=True,
            floatmode="fixed",
            prefix="       "
        )

        parts = [data_str]

        if self._op:
            parts.append(f"grad_fn=<{self._op}>")
        elif self.grad is not None and self.grad.any():
            parts.append("requires_grad=True")

        # Only show shape for tensors with more than 1 dimension
        if self.data.ndim > 1:
            parts.append(f"shape={self.shape}")

        parts.append(f"dtype={self.dtype}")

        joined = ", ".join(parts)
        return f"Tensor({joined})"


    #
    # Unary operations

    def __neg__(self):
        return self * -1

    def exp(self):
        data = np.exp(self.data)
        out = Tensor(data, _children=(self,), _op='exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        out = Tensor(np.log(self.data), _children=(self,), _op='log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward

        return out

    def t(self) -> 'Tensor':
        data = self.data.T
        out = Tensor(data, _children=(self, ), _op='transpose')

        def _backward():
            self.grad += out.grad.T
        out._backward = _backward

        return out

    def sigmoid(self) -> 'Tensor':
        data = 1 / (1 + np.exp(-self.data))
        out = Tensor(data, _children=(self,), _op='sigmoid')

        def _backward():
            self.grad += (out.data * (1.0 - out.data)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        data = np.maximum(self.data, 0)
        out = Tensor(data, _children=(self,), _op='ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        abs_x = np.abs(x)
        exp_neg_2_abs_x = np.exp(-2 * abs_x)
        
        # This formula is stable because exp() is only called on non-positive numbers
        data = np.sign(x) * (1 - exp_neg_2_abs_x) / (1 + exp_neg_2_abs_x)
        out = Tensor(data, _children=(self,), _op='tanh')
        
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward

        return out


    #
    # Binary operations

    def __add__(self, other):
        other = ensure_tensor(other, dtype=self.dtype)

        data = self.data + other.data
        out = Tensor(data, _children=(self, other), _op='+')

        def _backward():
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = ensure_tensor(other, dtype=self.dtype)
        return self + (-other)

    def __rsub__(self, other):
        other = ensure_tensor(other, dtype=self.dtype)
        return other + (-self)

    def __mul__(self, other):
        other = ensure_tensor(other, dtype=self.dtype)

        data = self.data * other.data
        out = Tensor(data, _children=(self, other), _op='*')

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1
    
    def __floordiv__(self, other):
        other = ensure_tensor(other, dtype=self.dtype)
        data = self.data // other.data
        out = Tensor(data, _children=(self, other), _op='//')

        def _backward():
            # Floor division has zero gradient almost everywhere (discontinuous),
            # so gradients are zeros — we simply don't accumulate anything.
            pass
        out._backward = _backward

        return out

    def __rfloordiv__(self, other):
        other = ensure_tensor(other, dtype=self.dtype)
        return other // self
    
    def __mod__(self, other):
        other = ensure_tensor(other, dtype=self.dtype)
        data = self.data % other.data
        out = Tensor(data, _children=(self, other), _op='%')

        def _backward():
            # d/dx (x % y) = 1 w.r.t x, -floor(x/y) w.r.t y
            self.grad  += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(
                -np.floor(self.data / other.data) * out.grad, other.shape
            )
        out._backward = _backward

        return out

    def __rmod__(self, other):
        other = ensure_tensor(other, dtype=self.dtype)
        return other % self
    
    def __pow__(self, other: int | float):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        data = self.data ** other
        out = Tensor(data, _children=(self,), _op=f'**{other}')

        # def _backward():
        #     self.grad += (other * self.data**(other-1)) * out.grad
        # out._backward = _backward

        def _backward():
            # Guard against 0^negative: where base is 0 the true gradient is 0,
            # but 0**(other-1) would produce inf/nan, so mask those positions.
            base_grad = other * np.where(
                self.data == 0, 0, self.data ** (other - 1)
            )
            self.grad += base_grad * out.grad
        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = ensure_tensor(other, dtype=self.dtype)

        data = self.data @ other.data
        out = Tensor(data, _children=(self, other), _op='@')

        # def _backward():
        #     self.grad += out.grad @ other.data.T
        #     other.grad += self.data.T @ out.grad
        # out._backward = _backward

        def _backward():
            # Works for both 2-D and batched (N-D) matmul.
            # For 2-D:  dL/dA = dL/dC @ B^T,  dL/dB = A^T @ dL/dC
            # For N-D:  same rule, just swap the last two axes for the transpose.
            self.grad  += out.grad @ other.data.swapaxes(-1, -2)
            other.grad += self.data.swapaxes(-1, -2) @ out.grad
        out._backward = _backward

        return out
    
    def matmul(self, other):
        return self @ other

    #
    # Reduce operations
    
    def sum(self, axis=None, keepdims=False):
        data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(data, _children=(self,), _op='sum')

        def _backward():
            grad = out.grad
            if not keepdims:
                # re-insert the collapsed axes so broadcast_to works
                if axis is None:
                    grad = np.full(self.shape, grad)
                else:
                    grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int] = None, keepdims: bool = False) -> 'Tensor':
        data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(data, _children=(self,), _op='mean')

        def _backward():
            grad = out.grad
            # Calculate scale (number of elements averaged over)
            if axis is None:
                scale = self.data.size
            else:
                axes = (axis,) if isinstance(axis, int) else axis
                scale = np.prod([self.data.shape[ax] for ax in axes])

            if not keepdims:
                if axis is None:
                    grad = np.full(self.data.shape, grad)
                else:
                    grad = np.expand_dims(grad, axis=axis)

            self.grad += np.broadcast_to(grad, self.data.shape) / scale

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        data = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(data, _children=(self,), _op='max')

        def _backward():
            # Create a mask where the input matches the max value
            # We use out.data (broadcasted) to find which elements 'won'
            mask = (self.data == out.data) 
            # Divide by sum of mask to handle ties (optional, but stable)
            self.grad += mask * out.grad 
            
        out._backward = _backward
        return out


    #
    #

    def zero_grad(self):
        self.grad = np.zeros(self.data.shape, dtype=self.dtype)

    def backward(self):
        # Topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Go one variable at a time and apply the chain rule to get its gradient
        self.grad.fill(1)
        for v in reversed(topo):
            v._backward()


def rand(*shape, dtype=np.float32) -> Tensor:
    """Return a Tensor of the given shape filled with standard-normal samples."""
    data = np.random.randn(*shape).astype(dtype)
    return Tensor(data=data, dtype=dtype)

def ones(*shape, dtype=np.float32) -> Tensor:
    """Return a Tensor of the given shape filled with 1.0."""
    # Handles both ones(5, 5) and ones((5, 5))
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    
    data = np.ones(shape, dtype=dtype)
    return Tensor(data=data, dtype=dtype)

def zeros(*shape, dtype=np.float32) -> Tensor:
    """Return a Tensor of the given shape filled with 0.0."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
        
    data = np.zeros(shape, dtype=dtype)
    return Tensor(data=data, dtype=dtype)