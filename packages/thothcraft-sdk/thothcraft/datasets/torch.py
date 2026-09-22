"""Optional PyTorch adapter; importing the main SDK does not import torch."""
from __future__ import annotations

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError as exc:
    raise ImportError('Install thothcraft-sdk[dl]') from exc


class ThothTorchDataset(TorchDataset):
    """Lazy per-minute tensors. Labels must be numeric, or use target_transform."""
    def __init__(self, dataset, sensor='radar', label='label',
                 transform=None, target_transform=None):
        self.dataset = dataset
        self.sensor = sensor
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        minute = self.dataset.minutes[index]
        x = minute[self.sensor].to_torch(dtype=torch.float32)
        y = minute.labels[self.label]
        return (self.transform(x) if self.transform else x,
                self.target_transform(y) if self.target_transform else torch.as_tensor(y))
