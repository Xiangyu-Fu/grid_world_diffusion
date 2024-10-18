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
from grid_world.common.grid_world_dataset_loader import PathDataset
from grid_world.common.diffusion_policy import DiffusionPolicy

# tqdm and wandb
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_diffusion_policy(policy, dataset, device, sample_index=None):
    # 如果没有提供 sample_index，随机选取一个
    if sample_index is None:
        sample_index = np.random.randint(0, len(dataset))

    print(f"Evaluating Sample Index: {sample_index}")
    sample = dataset[sample_index]
    observation_env = sample['observation.env'].unsqueeze(0).to(device)  # Add batch dimension
    observation_state = sample['observation.state'].unsqueeze(0).to(device)

    with torch.inference_mode():
        predicted_actions = policy.select_action(sample)
        print(f"Predicted Actions are: {predicted_actions}")

    # Plot the environment and the predicted trajectory
    batch_index = 0
    time_step = 0

    env_map = observation_env[batch_index, time_step]
    env_map = env_map.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots()
    ax.imshow(env_map)
    ax.scatter(observation_state[0, :, 0].cpu(), observation_state[0, :, 1].cpu(), c='green', label='current State')
    ax.plot(predicted_actions[:, 0].cpu(), predicted_actions[:, 1].cpu(), c='red', label='Predicted Trajectory')
    ax.set_title(f'Grid World Play - Sample Index {sample_index}')
    ax.legend()
    ax.grid(True)

    return fig  # 返回图像对象


def train_grid_world(resume_training=False, model_path='grid_world/models/diffusion_policy.pth'):
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

    observations = example_batch["observation.env"]         # (batch_size, num_envs, channel, observation_x, observation_y) --> [64, 2, 3, 100, 100]
    states = example_batch["observation.state"]             # (batch_size, num_states, 2) --> [64, 2, 2]
    actions = example_batch["action"]                       # (batch_size, num_actions, 2) --> [64, 15, 2]

    print(f"Observations shape: {observations.shape}")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")

    policy = DiffusionPolicy(dataset_stats=dataset.stats)
    policy.to(device)

    # Check if we want to resume training and if the model checkpoint exists
    if resume_training and os.path.exists(model_path):
        print(f"Loading model from {model_path} to resume training...")
        policy.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")

    policy.train()  # Set model to train mode

    training_steps = 50000

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Run training loop.
    step = 0
    best_loss = float('inf') 
    save_interval = 5000 
    best_model_path = 'grid_world/models/best_diffusion_policy.pth'  

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

                # # 每 100 步评估一次模型
                # if step % 100 == 0:
                #     policy.eval() 
                #     policy.reset()
                #     # fig = evaluate_diffusion_policy(policy, dataset, device)
                #     # wandb.log({"evaluation_plot": wandb.Image(fig), "step": step})
                #     # plt.close(fig)
                #     policy.train()

                if step % save_interval == 0:
                    model_path = f'grid_world/models/diffusion_policy_step_{step}.pth'
                    torch.save(policy.state_dict(), model_path)
                    # print(f"Model saved at step {step} to {model_path}")

                if loss.item() < best_loss and step % 1000 == 0:
                    best_loss = loss.item()
                    torch.save(policy.state_dict(), best_model_path)
                    # print(f"Best model saved at step {step} with loss {best_loss}")

            if step >= training_steps:
                print("Training steps reached. Training is complete!")
                break

    wandb.finish()

    # # 最终保存模型并命名为带时间戳的模型
    # model_timestamp = str(int(torch.cuda.Event(enable_timing=True).elapsed_time() / 1000))
    # final_model_path = f'grid_world/models/diffusion_policy_{model_timestamp}.pth'
    # torch.save(policy.state_dict(), final_model_path)
    # print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    train_grid_world(resume_training=False)
    print("Training Done!")