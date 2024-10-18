from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
from huggingface_hub import snapshot_download
from gymnasium import spaces

from grid_world.common.diffusion_policy import DiffusionPolicy

# Create a simple custom grid world environment
class GridWorldEnv(gym.Env):
    def __init__(self, grid_size=5, max_episode_steps=300):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Define action and observation space
        # Actions: agent position (x, y) directly
        self.action_space = spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)
        # Observation: agent position (x, y) and a pixel representation of the grid
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32),
            "pixels": spaces.Box(low=0, high=255, shape=(grid_size, grid_size, 3), dtype=np.uint8),
        })

        # Initialize agent position
        self.agent_pos = np.array([0, 0], dtype=np.int32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        return self._get_obs(), {}

    def step(self, action):
        # Update agent position based on action
        self.agent_pos = np.clip(action, 0, self.grid_size - 1)

        self.current_step += 1
        terminated = np.array_equal(self.agent_pos, [self.grid_size - 1, self.grid_size - 1])
        truncated = self.current_step >= self.max_episode_steps
        reward = 1 if terminated else -0.1

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        # Create a simple RGB representation of the grid
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        grid[self.agent_pos[1], self.agent_pos[0]] = [255, 0, 0]  # Red agent
        return grid

    def _get_obs(self):
        return {
            "agent_pos": self.agent_pos,
            "pixels": self.render(),
        }

# Create a directory to store the video of the evaluation
output_directory = Path("outputs/eval/example_grid_world_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

policy = DiffusionPolicy()
model_path='grid_world/models/diffusion_policy.pth'
policy.load_state_dict(torch.load(model_path))
policy.load_state_dict(torch.load(model_path))
policy.eval()

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Device set to:", device)
else:
    device = torch.device("cpu")
    print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")
    # Decrease the number of reverse-diffusion steps (trades off a bit of quality for 10x speed)
    policy.diffusion.num_inference_steps = 10

policy.to(device)

# Initialize evaluation environment
env = GridWorldEnv(grid_size=100, max_episode_steps=300)

# Reset the policy and environment to prepare for rollout
policy.reset()
numpy_observation, info = env.reset(seed=42)

# Prepare to collect every rewards and all the frames of the episode,
# from initial state to final state.
rewards = []
frames = []

# Render frame of the initial state
frames.append(env.render())

step = 0
done = False
while not done:
    # Prepare observation for the policy running in PyTorch
    state = torch.from_numpy(numpy_observation["agent_pos"])
    image = torch.from_numpy(numpy_observation["pixels"])

    # Convert to float32 with image from channel first in [0,255] to channel last in [0,1]
    state = state.to(torch.float32)
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)

    # Send data tensors from CPU to GPU
    state = state.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    # Create the policy input dictionary
    observation = {
        "observation.state": state,
        "observation.env": image,
    }

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        action = policy.select_action(observation)

    # Prepare the action for the environment
    numpy_action = action.squeeze(0).to("cpu").numpy()

    # Step through the environment and receive a new observation
    numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
    print(f"{step=} {reward=} {terminated=}")

    # Keep track of all the rewards and frames
    rewards.append(reward)
    frames.append(env.render())

    # The rollout is considered done when the success state is reach (i.e. terminated is True),
    # or the maximum number of iterations is reached (i.e. truncated is True)
    done = terminated or truncated or done
    step += 1

if terminated:
    print("Success!")
else:
    print("Failure!")

# Get the speed of environment (i.e. its number of frames per second).
fps = 10

# Encode all frames into a mp4 video.
video_path = output_directory / "rollout.mp4"
imageio.mimsave(str(video_path), np.stack(frames), fps=fps)

print(f"Video of the evaluation is available in '{video_path}'.")