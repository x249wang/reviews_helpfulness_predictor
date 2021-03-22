import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, feature_data, labels):
        self.feature_data = feature_data
        self.labels = labels
        self.n = len(labels)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.feature_data[index, :].astype(np.float32)),
            torch.tensor(self.labels[index]),
        )

    def __len__(self):
        return self.n
