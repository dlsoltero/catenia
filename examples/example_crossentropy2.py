import catenia.nn as nn
from catenia import Tensor
from catenia.optim import AdamW


# Input (Batch, Features)
x = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])

# Targets (Batch, Classes)
y_one_hot = Tensor([
    [1, 0, 0], # Class 0
    [0, 0, 1], # Class 2
    [0, 1, 0], # Class 1
    [1, 0, 0]  # Class 0
])

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 3)
)
optim = AdamW(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

for epoch in range(30):
    logits = model(x)
    loss = criterion(logits, y_one_hot)
    
    optim.zero_grad()
    loss.backward()
    optim.step()
        
    print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
