import numpy as np

import catenia
import catenia.nn as nn
from catenia.optim import AdamW
# from catenia.utils.data import TensorDataset, DataLoader
from catenia.lightning import Trainer

from datasets.mnist.idx import IdxFile


class MNISTNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Input: (batch, 1, 28, 28)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # (batch, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (batch, 32, 7, 7)
            nn.Flatten(),                                           # (batch, 32*7*7)
            nn.Linear(32 * 7 * 7, 10)                               # (batch, 10 classes)
        )

    def forward(self, x):
        return self.net(x)


# Preprocess the data

x_train = IdxFile('../datasets/mnist/train-images-idx3-ubyte.gz').read_all()
x_test = IdxFile('../datasets/mnist/t10k-images-idx3-ubyte.gz').read_all()
y_train = IdxFile('../datasets/mnist/train-labels-idx1-ubyte.gz').read_all()
y_test = IdxFile('../datasets/mnist/t10k-labels-idx1-ubyte.gz').read_all()

# Standardize
x_train_std = (x_train - x_train.mean()) / x_train.std()
x_test_std = (x_test - x_test.mean()) / x_test.std()

# Adjust shape
x_train_std = x_train_std[:, np.newaxis, :, :]
# x_test_std = x_test_std.reshape(-1, 28*28)

num_classes = 10
y_train_oh = np.eye(num_classes)[y_train.flatten().astype(int)]
# y_test_oh = np.eye(num_classes)[y_test.flatten().astype(int)]

# # Make tensors
# x_train_std = catenia.Tensor(x_train_std)
# y_train_oh = catenia.Tensor(y_train_oh)

# x_test_std = catenia.Tensor(x_test_std)
# # y_test_oh = catenia.Tensor(y_test_oh)


# dataset = TensorDataset(x_train_std, y_train_oh)
# loader = DataLoader(dataset=dataset, batch_size=32)

model = MNISTNet()
optimizer = AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

trainer = Trainer(model, optimizer, criterion)
# X_train shape should be (N, Channels, H, W)
trainer.fit(x_train_std, y_train_oh, epochs=4, batch_size=32)

catenia.save(model.state_dict(), 'cnn_mnist.ctn')
