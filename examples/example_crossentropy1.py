import catenia.nn as nn
from catenia import Tensor
from catenia.optim import AdamW


# Input (Batch, Features)
x = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])

# Targets (Batch,) -> Integer indices representing the correct class
y_indices = Tensor([0, 2, 1, 0]) 

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 3)
)
optim = AdamW(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

for epoch in range(30):
    logits = model(x)
    loss = criterion(logits, y_indices)
    
    optim.zero_grad()
    loss.backward()
    optim.step()
        
    print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
