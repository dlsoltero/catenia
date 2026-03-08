from abc import ABC, abstractmethod

import numpy as np

from catenia.tensor import Tensor


class Dataset(ABC):

    @abstractmethod
    def __getitem__(self, index: int):
        """Fetch a single data point and its label"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples"""
        pass

class TensorDataset(Dataset):
    """A simple dataset wrapper for tensors (X and y)"""

    def __init__(self, *tensors: Tensor):
        # Ensure all tensors have the same first dimension (number of samples)
        assert all(tensors[0].shape[0] == t.shape[0] for t in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]


class DataLoader:

    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(dataset)
        self.indices = np.arange(self.n_samples)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            yield self._collate_fn(batch_indices)

    def _collate_fn(self, indices):
        """
        Gathers individual samples into a batch.
        In a real framework, this handles converting lists of arrays into a single Tensor.
        """
        samples = [self.dataset[i] for i in indices]
        
        # Zip (*samples) unzips [(x1, y1), (x2, y2)] into ([x1, x2], [y1, y2])
        return tuple(self._stack_tensors(items) for items in zip(*samples))

    def _stack_tensors(self, items):
        # Convert list of NumPy/Tensor data into a single batch Tensor
        # We assume your Tensor class can take a NumPy array of the stacked data
        data = np.array([item.data if isinstance(item, Tensor) else item for item in items])
        # Note: You'll need to pass the correct dtype here from your framework
        return Tensor(data)
