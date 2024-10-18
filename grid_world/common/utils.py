import torch
import torch.nn as nn
from torch import Tensor

class Normalize(nn.Module):
    def __init__(self, min_max_values: dict[str, tuple[float, float]]):
        """
        Args:
            min_max_values: A dictionary where keys are the names of the features to normalize,
                            and values are tuples containing (min_value, max_value) for each feature.
        """
        super().__init__()
        self.min_max_values = min_max_values

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        normalized_batch = {}
        for key, value in batch.items():
            if key in self.min_max_values:
                min_val, max_val = self.min_max_values[key]
                # Ensure min_val and max_val are tensors with the same device and dtype as value
                min_val = torch.tensor(min_val, device=value.device, dtype=value.dtype)
                max_val = torch.tensor(max_val, device=value.device, dtype=value.dtype)
                # Avoid division by zero
                denom = max_val - min_val
                denom[denom == 0] = 1.0
                # Normalize
                normalized_value = (value - min_val) / denom
                normalized_batch[key] = normalized_value
            else:
                normalized_batch[key] = value
        return normalized_batch

class Unnormalize(nn.Module):
    def __init__(self, min_max_values: dict[str, tuple[float, float]]):
        """
        Args:
            min_max_values: A dictionary where keys are the names of the features to unnormalize,
                            and values are tuples containing (min_value, max_value) for each feature.
        """
        super().__init__()
        self.min_max_values = min_max_values

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        unnormalized_batch = {}
        for key, value in batch.items():
            if key in self.min_max_values:
                min_val, max_val = self.min_max_values[key]
                # Ensure min_val and max_val are tensors with the same device and dtype as value
                min_val = torch.tensor(min_val, device=value.device, dtype=value.dtype)
                max_val = torch.tensor(max_val, device=value.device, dtype=value.dtype)
                # Unnormalize
                unnormalized_value = value * (max_val - min_val) + min_val
                unnormalized_batch[key] = unnormalized_value
            else:
                unnormalized_batch[key] = value
        return unnormalized_batch
