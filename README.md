# Catenia

A autograd engine and neural networks library.

# Name

Derived from the Latin Catena, meaning "Chain"

# Tests

`pytest`
`pytest tests/test_tensor.py`

# Notes

- Gradient accumulation is required for correct backpropagation through shared nodes

When a tensor appears more than once in a computation graph, its gradient contributions from each use must be summed, not overwritten.

```python
# Wrong — silently drops gradient from all but the last path
def _backward():
    self.grad = out.grad * other.data   # overwrites

# Correct — accumulates contributions from every path
def _backward():
    self.grad += out.grad * other.data  # accumulates
```