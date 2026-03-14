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
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self) -> str:
        return self.data.dtype

    def __len__(self) -> int:
        # If the data is a scalar (0-d), len() should technically 
        # raise a TypeError, but NumPy usually handles this via shape.
        if self.data.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return len(self.data)

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

    def __getstate__(self):
        """Define what gets saved. Exclude non-serializable functions."""
        state = self.__dict__.copy()
        state['_backward'] = None
        state['_prev'] = set()
        state['_op'] = ''
        return state

    def __setstate__(self, state):
        """Define how the object is restored."""
        self.__dict__.update(state)
        if '_backward' not in state or state['_backward'] is None:
            self._backward = lambda: None

    def __getitem__(self, index):
        data = self.data[index]
        out = Tensor(data, _children=(self,), _op='getitem')

        def _backward():
            # We need to flow the gradient back
            # only to the indices that were sliced.
            # We create a zero-filled array of the same shape as the parent
            # and add the 'out' gradient to the specific index.
            np.add.at(self.grad, index, out.grad)

        out._backward = _backward
        return out

    def reshape(self, *shape) -> 'Tensor':
        # Handle both reshape(1, 10) and reshape((1, 10))
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]

        out = Tensor(self.data.reshape(shape), _children=(self,), _op='reshape')

        def _backward():
            # The gradient for the input is the output gradient 
            # reshaped back to the input's original shape
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward

        return out

    def gather_nd(self, indices: 'Tensor') -> 'Tensor':
        """
        Gathers values from the tensor along the last axis using integer indices.
        Assumes self is (Batch, Classes) and indices is (Batch,).
        """
        # Ensure indices are integers for indexing
        idx_data = indices.data.astype(int)
        batch_range = np.arange(self.data.shape[0])

        # Forward pass: extract log-probs of the true class
        out_data = self.data[batch_range, idx_data]
        out = Tensor(out_data, _children=(self,), _op='gather')

        def _backward():
            # Backward pass: scatter the gradient back only to the selected indices
            # np.add.at is used to handle duplicate indices correctly
            np.add.at(self.grad, (batch_range, idx_data), out.grad)
        out._backward = _backward

        return out


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

    def leaky_relu(self, alpha=0.01):
        data = np.where(self.data > 0, self.data, self.data * alpha)
        out = Tensor(data, _children=(self,), _op='LeakyReLU')

        def _backward():
            # Gradient is 1 if x > 0, and alpha if x <= 0
            mask = np.where(self.data > 0, 1.0, alpha)
            self.grad += mask * out.grad
        out._backward = _backward

        return out

    def elu(self, alpha=1.0):
        data = np.where(self.data > 0, self.data, alpha * (np.exp(self.data) - 1))
        out = Tensor(data, _children=(self,), _op='ELU')

        def _backward():
            # Gradient is 1 if x > 0, and alpha * exp(x) (which is data + alpha) if x <= 0
            grad_map = np.where(self.data > 0, 1.0, out.data + alpha)
            self.grad += grad_map * out.grad
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

    def softmax(self, axis=-1) -> 'Tensor':
        # Subtract max for numerical stability (prevents exp overflow)
        max_val = self.data.max(axis=axis, keepdims=True)
        exps = np.exp(self.data - max_val)
        sum_exps = exps.sum(axis=axis, keepdims=True)

        data = exps / sum_exps
        out = Tensor(data, _children=(self,), _op='softmax')

        def _backward():
            # Softmax gradient: dS_i/dx_j = S_i(delta_ij - S_j)
            # This is more efficient if handled during CrossEntropy,
            # but for a general implementation:
            for i, (s, grad) in enumerate(zip(out.data, out.grad)):
                s = s.reshape(-1, 1)
                # Jacobian matrix: diag(s) - s @ s.T
                jacobian = np.diagflat(s) - (s @ s.T)
                self.grad[i] += (jacobian @ grad.reshape(-1, 1)).reshape(self.shape[1:])
        # Note: The loop-based backward is slow for large batches.
        # Usually, Softmax is paired with CrossEntropy for a simplified gradient.
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

    def conv2d(self, weight: 'Tensor', bias: 'Tensor' = None, stride=1, padding=0) -> 'Tensor':
            N, C, H, W = self.shape
            F, _, HH, WW = weight.shape
            out_h = (H + 2 * padding - HH) // stride + 1
            out_w = (W + 2 * padding - WW) // stride + 1

            x_padded = np.pad(self.data, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')

            # Vectorized window extraction (im2col)
            shape = (N, C, HH, WW, out_h, out_w)
            strides = (
                x_padded.strides[0], x_padded.strides[1],
                x_padded.strides[2], x_padded.strides[3],
                x_padded.strides[2] * stride, x_padded.strides[3] * stride
            )
            # x_cols shape: (N, C, HH, WW, out_h, out_w)
            x_cols = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)

            # weight: (F, C, HH, WW) -> 'fcij'
            # x_cols: (N, C, HH, WW, out_h, out_w) -> 'ncijhw'
            # output: (N, F, out_h, out_w) -> 'nfhw'
            out_data = np.einsum('fcij,ncijhw->nfhw', weight.data, x_cols)

            if bias is not None:
                out_data += bias.data.reshape(1, -1, 1, 1)

            out = Tensor(out_data, _children=(self, weight, bias) if bias is not None else (self, weight), _op='conv2d')

            def _backward():
                # dL/dBias: Sum over batch, height, and width
                if bias is not None:
                    bias.grad += out.grad.sum(axis=(0, 2, 3))

                # dL/dWeight: 'nfhw' (grad) * 'ncijhw' (cols) -> 'fcij' (weight grad)
                weight.grad += np.einsum('nfhw,ncijhw->fcij', out.grad, x_cols)

                # dL/dx (vectorized): Calculate gradient for the padded input
                # We use a 'col2im' approach by back-broadcasting the gradient
                # into the window shapes and then accumulating.
                d_padded = np.zeros_like(x_padded)

                # Create a view of d_padded to accumulate gradients into
                d_padded_cols = np.lib.stride_tricks.as_strided(d_padded, shape=shape, strides=strides)

                # Backpropagate through the weights
                # This is essentially: d_padded_cols += weight.T @ out.grad
                # dL/dx: 'fcij' (weight) * 'nfhw' (grad) -> 'ncijhw' (padded input grad)
                np.einsum('fcij,nfhw->ncijhw', weight.data, out.grad, out=d_padded_cols, casting='same_kind')

                # Remove padding to return to original input shape
                if padding > 0:
                    self.grad += d_padded[:, :, padding:-padding, padding:-padding]
                else:
                    self.grad += d_padded
            out._backward = _backward

            return out

    def binary_cross_entropy(self, target: 'Tensor') -> 'Tensor':
            """
            Computes Binary Cross Entropy (BCE): -[y*log(p) + (1-y)*log(1-p)]
            Assumes 'self' are probabilities (output of sigmoid)
            and 'target' are single values {0,1}
            """
            target = ensure_tensor(target, dtype=self.dtype)

            #  Clip predictions to avoid log(0) or log(1)
            eps = 1e-12
            clipped_data = np.clip(self.data, eps, 1.0 - eps)

            # Create a child tensor for the clipped data
            # We use a simple identity-style backward for the clipping operation
            p = Tensor(clipped_data, _children=(self,), _op='clip')
            def _clip_backward():
                self.grad += p.grad
            p._backward = _clip_backward

            # Calculate BCE using existing tensor operations
            # L = -(y * log(p) + (1 - y) * log(1 - p))
            term1 = target * p.log()
            term2 = (1.0 - target) * (1.0 - p).log()

            # Final loss is the negative mean of the combined terms
            return (term1 + term2).mean() * -1.0

    def categorical_cross_entropy(self, target: 'Tensor') -> 'Tensor':
            """
            Computes Categorical Cross Entropy loss.
            Assumes 'self' is the output of a softmax (probabilities)
            and 'target' is one-hot encoded.
            """
            target = ensure_tensor(target, dtype=self.dtype)

            # Clip self.data to avoid log(0) or log(1)
            # We create a new tensor with clipped data to prevent NaN gradients
            eps = 1e-12
            clipped_data = np.clip(self.data, eps, 1.0 - eps)
            clipped_self = Tensor(clipped_data, _children=(self,), _op='clip')

            # Manually define the clipping backward (identity-ish)
            def _clip_backward():
                self.grad += clipped_self.grad
            clipped_self._backward = _clip_backward

            # Loss = -sum(target * log(clipped_probs))
            log_probs = clipped_self.log()
            elementwise_loss = target * log_probs

            # Sum over classes (axis -1), then mean over the batch
            return elementwise_loss.sum().mean() * -1.0

    def cross_entropy(self, target: 'Tensor', axis=-1, reduction: str | None = 'mean') -> 'Tensor':
        """
        Computes Categorical Cross Entropy (Stable LogSoftmax + NLL).
        Handles both one-hot targets (Batch, Classes) and integer-index targets (Batch,).
        """
        target = ensure_tensor(target, dtype=self.dtype)

        # Stable LogSoftmax logic
        max_val = self.max(axis=axis, keepdims=True)
        shifted_logits = self - max_val
        log_sum_exp = shifted_logits.exp().sum(axis=axis, keepdims=True).log()
        log_probs = shifted_logits - log_sum_exp

        # Selection Logic (NLL)
        if self.shape == target.shape:
            # Case A: target is one-hot (Batch, Classes)
            loss = (target * log_probs).sum(axis=axis) * -1.0
        else:
            # Case B: target is integer indices (Batch,)
            # Use our new gather_nd to pick the correct class log-probs
            loss = log_probs.gather_nd(target) * -1.0

        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()

        return loss


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
    """
    Return a Tensor of the given shape filled with numbers from a
    uniform distribution on the interval [0,1].
    """
    # Handles both ones(5, 5) and ones((5, 5))
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]

    data = np.random.rand(*shape).astype(dtype)
    return Tensor(data=data, dtype=dtype)

def ones(*shape, dtype=np.float32) -> Tensor:
    """Return a Tensor of the given shape filled with 1.0."""
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

def save(obj, f):
    """Saves a dictionary (or any object) to a file."""
    import pickle
    with open(f, 'wb') as opened_file:
        pickle.dump(obj, opened_file)

def load(f):
    """Loads an object and maps tensors to the specified device."""
    import pickle
    with open(f, 'rb') as opened_file:
        obj = pickle.load(opened_file)
        return obj
