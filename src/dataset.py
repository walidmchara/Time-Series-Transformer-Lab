from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def make_sequences(x, y, sequence_length):
    xs, ys = [], []
    for end in range(sequence_length - 1, len(x)):
        start = end - sequence_length + 1
        xs.append(x[start:end + 1])
        ys.append(y[end])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
