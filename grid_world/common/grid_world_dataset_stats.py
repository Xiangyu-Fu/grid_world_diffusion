# calculate the statistics of the dataset and save to a .pkl file
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from grid_world.common.grid_world_dataset_loader import PathDataset
from grid_world.common.diffusion_policy import DiffusionPolicy
import numpy as np
import pickle
import os
import gymnasium as gym
from gymnasium import spaces


def calculate_grid_world_stats():
    # delta_timestamps
    delta_timestamps = {
        "observation.env": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    dataset = PathDataset('grid_world/dataset/grid_world_dataset.pkl', delta_timestamps=delta_timestamps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Calculate the statistics of the dataset
    stats = dataset.calculate_stats()

    stats = dataset.stats


    # Save the statistics to a .pkl file
    with open('grid_world/dataset/grid_world_dataset_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)


if __name__ == "__main__":
    calculate_grid_world_stats()