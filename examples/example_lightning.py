import pandas as pd

from catenia.nn import Module, Linear, MSELoss
from catenia.utils.data import random_train_test_split
from catenia.optim import AdamW
from catenia.lightning import Trainer


# Preprocess the data

df = pd.read_csv('../datasets/boston/boston.csv')

features = ["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "TAX", "PTRATIO", "B", "LSTAT"]  # CHAS, RAD
X = df[features]
y =  df.loc[:, ["MEDV"]]

x_train, x_test, y_train, y_test = random_train_test_split(X.to_numpy(), y.to_numpy(), random_state=1)

# Standardize
x_train_std = (x_train - x_train.mean()) / x_train.std()
x_test_std = (x_test - x_test.mean()) / x_test.std()


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


model = MLP()
optimizer = AdamW(model.parameters(), lr=0.001)
criterion = MSELoss()

trainer = Trainer(model, optimizer, criterion)

trainer.fit(
    train_data=x_train_std,
    train_target=y_train,
    epochs=50,
    batch_size=32,
    val_data=x_test_std,
    val_target=y_test,
    patience=1
)
