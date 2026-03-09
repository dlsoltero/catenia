# Catenia

**Catenia** is a lightweight, educational deep learning framework built from scratch using NumPy. It implements an autograd engine and a neural networks library, similar to PyTorch, providing a transparent look at how modern neural networks function under the hood. It was initially inspired by [micrograd](https://github.com/karpathy/micrograd).


## Installation

You can install **Catenia** directly from this source.

### 1. Clone the Repository

```bash
git clone https://github.com/dlsoltero/catenia.git
cd catenia
```

### 2. Install via pip

It is recommended to use a virtual environment:

```bash
# Create and activate a virtual environment
python3 -m venv env
source env/bin/activate

# Install the package
pip install -e .
```

> [!NOTE]
> The `-e` (editable) flag allows you to make changes to the source code (like modifying `tensor.py` or `nn.py`) and have them reflected immediately without reinstalling. Remove if this is not required.


## Framework Architecture & Features

Catenia is organized into four modular pillars, separating the low-level autograd engine from high-level training utilities.

- Autograd Engine (`tensor.py`)
- Neural Modules (`nn.py`)
- Optimization Suite (`optim.py`)
- Training Logic (`lightning.py`)


## Usage Example

Catenia offers two ways to train: a manual loop for complete control over the training process, and a Trainer to eliminate boilerplate code.


```python
import numpy as np

import catenia.nn as nn
from catenia import Tensor
from catenia.tensor import Tensor
from catenia.optim import AdamW
from catenia.utils.data import random_train_test_split

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        ])

    def forward(self, x):
        out = self.layers[0](x).relu()
        out = self.layers[1](out).relu()
        return self.layers[2](out)

# Generate non-linear synthetic data
np.random.seed(42)
X = np.linspace(-5, 5, 500).reshape(-1, 1)
y = (X**2 + 2*X + 1 + np.random.normal(0, 1, X.shape))

x_train, x_val, y_train, y_val = random_train_test_split(X, y, random_state=1)


model = MLP(input_dim=1, hidden_dim=32, output_dim=1)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = nn.MSELoss()
```

Then you can do manual training (the traditional way):

```python
from catenia.utils.data import TensorDataset, DataLoader

x_train = Tensor(x_train)
y_train = Tensor(y_train)
x_val = Tensor(x_val)
y_val = Tensor(y_val)

dataset = TensorDataset(x_train, y_train)
loader = DataLoader(dataset=dataset, batch_size=32)

for epoch in range(200):
    epoch_loss = 0
    batch_count = 0
    for batch in loader:
        xb, yb = batch

        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data
        batch_count += 1

    avg_loss = epoch_loss / batch_count

    # Validation
    val_pred = model(x_val)
    val_loss = criterion(val_pred, y_val).data

    print(f"epoch {epoch:3d}, train loss: {avg_loss:.4f}, val loss: {val_loss:.4f}")
```

or you can do automated training:

```python
from catenia.lightning import Trainer

trainer = Trainer(model, optimizer, criterion)

trainer.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    val_data=x_val, 
    val_target=y_val,
    patience=3
)
```


## License

This project is licensed under the MIT License. It is intended for educational purposes.
