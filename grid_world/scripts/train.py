# Hydra app
# import hydra
# from omegaconf import DictConfig, OmegaConf

# Torch
import torch
# from torch import nn, Tensor
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
# from PIL import Image
# import torchvision
# from torchviz import make_dot
# from torchvision.transforms import ToPILImage

# Python
# import math
# import numpy as np
# import einops
# from pathlib import Path
# from collections import deque

# import sys
# import os

# grid_world
from grid_world.common.grid_world_data_loader import PathDataset
from grid_world.common.diffusion_policy import DiffusionPolicy

# tqdm and wandb
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np

def train_grid_world():
    # Initialize wandb
    wandb.init(project="grid_world_training", name="train_diffusion_policy")

    # delta_timestamps
    delta_timestamps = {
        "observation.env": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    dataset = PathDataset('grid_world/dataset/grid_world_dataset.pkl', delta_timestamps=delta_timestamps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader
    dataloader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=64,
        shuffle=True, 
        pin_memory=device != torch.device("cpu"),
        drop_last=True, # Drop last batch if it's smaller than batch_size
    )

    print(f"Dataset size: {len(dataset)}")

    example_batch = next(iter(dataloader))
    print("Example batch keys:", example_batch.keys())

    observations = example_batch["observation.env"]      # (batch_size, num_envs, channel, observation_x, observation_y) --> [64, 2, 3, 100, 100]
    states = example_batch["observation.state"]  # (batch_size, num_states, 2) --> [64, 2, 2]
    actions = example_batch["action"]            # (batch_size, num_actions, 2) --> [64, 15, 2]

    print(f"Observations shape: {observations.shape}")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")

    policy = DiffusionPolicy()
    policy.to(device)
    policy.train()

    training_steps = 5000

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Run training loop.
    step = 0
    with tqdm(total=training_steps, desc="Training Progress") as pbar:
        while step < training_steps:
            for batch in dataloader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}  # batch.items() returns key-value pairs
                output_dict = policy.forward(batch)
                loss = output_dict["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Log metrics to wandb
                wandb.log({"loss": loss.item(), "step": step})
                step += 1
                pbar.update(1)

                if step >= training_steps:
                    break

    wandb.finish()
    torch.save(policy.state_dict(), 'grid_world/models/diffusion_policy.pth')


if __name__ == "__main__":
    train_grid_world()
    print("Training Done!")