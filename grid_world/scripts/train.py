# Hydra app
# import hydra
from omegaconf import DictConfig, OmegaConf

# Torch
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from PIL import Image
import torchvision
from torchviz import make_dot
# from torchvision.transforms import ToPILImage

# Python
import math
import numpy as np
import einops
from pathlib import Path
from collections import deque

import sys
import os

# grid_world
from grid_world.common.grid_world_data_loader import PathDataset
from grid_world.common.diffusion_policy import DiffusionPolicy


def train_grid_world():
    # # print(OmegaConf.to_yaml(config))
    # output_directory = Path("outputs/test_outputs/example_pusht_diffusion")
    # output_directory.mkdir(parents=True, exist_ok=True)

    # 调整后的 delta_timestamps
    delta_timestamps = {
        "observation.state": [0.0],
        "action": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    # 创建数据集实例
    dataset = PathDataset('grid_world/dataset/grid_world_dataset.pkl', delta_timestamps=delta_timestamps)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=64,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    # 显示数据集大小
    print(f"Dataset size: {len(dataset)}")

    # 获取一个批次的数据
    example_batch = next(iter(dataloader))
    print("Example batch keys:", example_batch.keys())

    # 提取状态和动作
    states = example_batch["observation.state"]  # (batch_size, num_states, 2)
    actions = example_batch["action"]            # (batch_size, num_actions, 2)

    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")

    # 创建模型
    policy = DiffusionPolicy()
    policy.to(device)
    policy.train()

    # 训练参数
    training_steps = 5000
    device = torch.device("cuda")
    log_freq = 250

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}  # batch.items() 返回键值对
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break


if __name__ == "__main__":
    train_grid_world()
    print("Done!")