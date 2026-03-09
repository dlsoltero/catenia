import pandas as pd

import catenia
from catenia.nn import Module, Linear, MSELoss
from catenia.utils.data import random_train_test_split, TensorDataset, DataLoader
from catenia.optim import AdamW


# Preprocess the data

df = pd.read_csv('../datasets/boston/boston.csv')

features = ["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "TAX", "PTRATIO", "B", "LSTAT"]  # CHAS, RAD
X = df[features]
y =  df.loc[:, ["MEDV"]]

x_train, x_test, y_train, y_test = random_train_test_split(X.to_numpy(), y.to_numpy(), random_state=1)

# Standardize
x_train_std = (x_train - x_train.mean()) / x_train.std()
x_test_std = (x_test - x_test.mean()) / x_test.std()

# Make tensors
x_train_std = catenia.Tensor(x_train_std)
y_train = catenia.Tensor(y_train)

x_test_std = catenia.Tensor(x_test_std)
y_test = catenia.Tensor(y_test)


class MLP(Module):

    def __init__(self):
        super().__init__()

        self.layer1 = Linear(11, 20)
        self.layer2 = Linear(20, 5)
        self.layer3 = Linear(5, 1)

    def forward(self, x):
        out = self.layer1(x).relu()
        out = self.layer2(out).relu()
        out = self.layer3(out).relu()
        return out


dataset = TensorDataset(x_train_std, y_train)
loader = DataLoader(dataset=dataset, batch_size=32)

model = MLP()
optimizer = AdamW(model.parameters(), lr=0.001)
criterion = MSELoss()

for epoch in range(30):
    epoch_loss = 0
    batch_count = 0
    for batch in loader:
        batch_x, batch_y = batch

        pred = model(batch_x)
        loss = criterion(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss
        batch_count += 1

    avg_loss = (epoch_loss / batch_count).data.item()

    print(f"epoch {epoch+1}, loss: {avg_loss:0.6f}")


# Evaluate

test_dataset = TensorDataset(x_test_std, y_test)
loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))
x, y = next(iter(loader))

pred = model(x)

mse = ((pred - y)**2).mean().data.item()
print(f"test MSE: {mse:0.4f}")
