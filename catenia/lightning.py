import numpy as np

from catenia import Tensor


class Trainer:

    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = {'train_loss': [], 'val_loss': []}

    def _run_epoch(self, X, y, batch_size, training=True):
        n = X.shape[0]
        indices = np.arange(n)
        if training: np.random.shuffle(indices)

        total_loss = 0
        batches_count = 0
        for i in range(0, n, batch_size):
            idx = indices[i:i+batch_size]

            # Forward
            xb, yb = Tensor(X[idx]), Tensor(y[idx])
            preds = self.model(xb)
            loss = self.criterion(preds, yb)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.data
            batches_count += 1

        return total_loss / batches_count

    def fit(self, train_data, train_target, epochs, batch_size=32, val_data=None, val_target=None, patience=5):
        best_loss = float('inf')
        wait = 0

        for epoch in range(1, epochs + 1):
            # Training loop
            epoch_loss = self._run_epoch(train_data, train_target, batch_size, training=True)
            self.history['train_loss'].append(epoch_loss)

            # Validation / Monitoring
            status_msg = f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}"

            if val_data is not None:
                val_loss = self._run_epoch(val_data, val_target, batch_size, training=False)
                self.history['val_loss'].append(val_loss)
                status_msg += f", Val Loss: {val_loss:.4f}"

                # Early Breaking Logic
                if val_loss < best_loss:
                    best_loss = val_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"\n[*] Early stop due to overfitting at epoch {epoch}")
                        break

            print(status_msg)
