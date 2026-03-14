import numpy as np

from datasets.mnist.idx import IdxFile

import catenia
from catenia.nn import Module, Linear, CrossEntropyLoss
from catenia.optim import AdamW
from catenia.utils.data import TensorDataset, DataLoader


# Preprocess the data

x_train = IdxFile('../datasets/mnist/train-images-idx3-ubyte.gz').read_all()
x_test = IdxFile('../datasets/mnist/t10k-images-idx3-ubyte.gz').read_all()
y_train = IdxFile('../datasets/mnist/train-labels-idx1-ubyte.gz').read_all()
y_test = IdxFile('../datasets/mnist/t10k-labels-idx1-ubyte.gz').read_all()

# Standardize
x_train_std = (x_train - x_train.mean()) / x_train.std()
x_test_std = (x_test - x_test.mean()) / x_test.std()

# Adjust shape
x_train_std = x_train_std.reshape(-1, 28*28)
x_test_std = x_test_std.reshape(-1, 28*28)

num_classes = 10
y_train_oh = np.eye(num_classes)[y_train.flatten().astype(int)]
# y_test_oh = np.eye(num_classes)[y_test.flatten().astype(int)]

# Make tensors
x_train_std = catenia.Tensor(x_train_std)
y_train_oh = catenia.Tensor(y_train_oh)

x_test_std = catenia.Tensor(x_test_std)
# y_test_oh = catenia.Tensor(y_test_oh)


class MLP(Module):

    def __init__(self):
        super().__init__()

        self.layer1 = Linear(28*28, 1024)
        self.layer2 = Linear(1024, 512)
        self.layer3 = Linear(512, 10)

    def forward(self, x):
        out = self.layer1(x).relu()
        out = self.layer2(out).relu()
        out = self.layer3(out).relu()
        return out


dataset = TensorDataset(x_train_std, y_train_oh)
loader = DataLoader(dataset=dataset, batch_size=32)

model = MLP()
optimizer = AdamW(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss(reduction='mean')

do_training = True

if do_training:
    for epoch in range(10):
        epoch_loss = 0
        batch_count = 0
        for batch in loader:
            x, y = batch

            pred = model(x)
            loss = criterion(pred, y)

            model.zero_grad()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            batch_count += 1

        avg_loss = (epoch_loss / batch_count).data.item()

        print(f"epoch {epoch}, loss: {avg_loss:0.6f}")

    # Save model
    catenia.save(model.state_dict(), 'mnist.ctn')
else:
    # Load model
    state_dict = catenia.load('mnist.ctn')
    model.load_state_dict(state_dict)    


# Evaluate

test_dataset = TensorDataset(x_test_std, y_test)
loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))
x, y = next(iter(loader))

pred = model(x)
pred_labels = np.argmax(pred.data, axis=1)

acc = sum(y_test == pred_labels.data) / len(y_test)
print(f"Test acc: {acc}")
