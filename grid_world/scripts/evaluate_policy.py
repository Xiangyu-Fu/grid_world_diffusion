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


# Create a simple custom grid world environment
class GridWorldEnv(gym.Env):
    def __init__(self, data_path, delta_timestamps, grid_size=100, max_episode_steps=300):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.delta_timestamps = delta_timestamps
        with open(data_path, 'rb') as f:
            self.paths = pickle.load(f)

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
        self.obstacle_points = None
        self.path = None

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        path_index = np.random.randint(0, len(self.paths))
        path_data = self.paths[path_index]
        self.path = np.array(path_data['path'])
        start_position = self.path[0, 1:].astype(int)
        goal_position = self.path[-1, 1:].astype(int)
        print(f"Start position: {start_position}, Goal position: {goal_position}")
        self.obstacle_points = np.array(path_data['obstacle'])
        self.agent_pos = self.path[0, 1:].astype(np.int32)
        return self._get_obs(), {}

    def step(self, action):
        # Update agent position based on action
        self.agent_pos = np.clip(action, 0, self.grid_size - 1)

        self.current_step += 1
        terminated = np.array_equal(self.agent_pos, self.path[-1, 1:].astype(np.int32))
        truncated = self.current_step >= self.max_episode_steps
        reward = 1 if terminated else -0.1

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        # Create a simple RGB representation of the grid
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for point in self.obstacle_points:
            x, y = point
            grid[y, x] = [255, 255, 255]  # White obstacles
        grid[self.agent_pos[1], self.agent_pos[0]] = [255, 0, 0]  # Red agent
        start_position = self.path[0, 1:].astype(int)
        goal_position = self.path[-1, 1:].astype(int)
        grid[start_position[1], start_position[0]] = [0, 255, 0]  # Green start position
        grid[goal_position[1], goal_position[0]] = [0, 0, 255]  # Blue goal position
        return grid

    def _get_obs(self):
        return {
            "agent_pos": self.agent_pos,
            "pixels": self.render(),
        }

def evaluate_diffusion_policy_to_video(model_path, dataset_path, output_video_path, num_episodes=1):
    # Load the dataset
    delta_timestamps = {
        "observation.env": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }
    dataset = PathDataset(dataset_path, delta_timestamps=delta_timestamps)

    # Load the trained model
    policy = DiffusionPolicy(dataset_stats=dataset.stats)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.reset()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fps = 5
    writer = FFMpegWriter(fps=fps)

    # Initialize the environment
    gridWorldEnv = GridWorldEnv(data_path=dataset_path, delta_timestamps=delta_timestamps, grid_size=100, max_episode_steps=300)
    numpy_observation, info = gridWorldEnv.reset(seed=42)

    step = 0
    done = False
    episode = 0

    while not done:
        state = torch.from_numpy(numpy_observation["agent_pos"]).to(device)
        image = torch.from_numpy(numpy_observation["pixels"]).to(device)

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

        observation = {
            "observation.state": state,
            "observation.env": image,
        }

        image = image.squeeze(0).cpu().numpy()

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = policy.select_action(observation)
            
        numpy_action = action.squeeze(0).to("cpu").numpy()
        # int
        numpy_action = np.round(numpy_action).astype(int)

        # Step through the environment and receive a new observation
        step += 1
        numpy_observation, reward, terminated, truncated, info = gridWorldEnv.step(numpy_action)
        print(f"{step=} {reward=} {terminated=}{numpy_action=}")
        
        # print(f"Episode {episode + 1} - Sample Index: {index}")
        # sample = dataset[index]
        # observation_env = sample['observation.env'].unsqueeze(0).to(device)  # Add batch dimension
        # observation_state = sample['observation.state'].unsqueeze(0).to(device)

        # # Set up the plot
        # fig, ax = plt.subplots()
        # env_map = observation_env[0, 0].permute(1, 2, 0).cpu().numpy()
        # im = ax.imshow(env_map)
        # current_state_plot = ax.scatter([], [], c='green', label='current State')
        # predicted_trajectory_plot, = ax.plot([], [], c='red', label='Predicted Trajectory')
        # ax.set_title(f'Grid World Play - Episode {episode + 1}')
        # ax.legend()
        # ax.grid(True)

        # with writer.saving(fig, output_video_path.format(episode + 1), dpi=100):
        #     for step in range(len(sample['action'])):
        #         with torch.inference_mode():
        #             # Select action using the current state
        #             predicted_actions = policy.select_action(sample)

        #         # Execute the first action from predicted actions
        #         action_to_execute = predicted_actions[0].cpu().numpy()

        #         # Update current observation state with the executed action
        #         observation_state[0, 0] = torch.tensor(action_to_execute).to(device)

        #         # Update the plot with the new state and trajectory
        #         current_state_plot.set_offsets(observation_state[0, :, :2].cpu().numpy())
        #         predicted_trajectory_plot.set_data(predicted_actions[:, 0].cpu().numpy(),
        #                                            predicted_actions[:, 1].cpu().numpy())

        #         # Capture the current frame
        #         writer.grab_frame()

        # plt.close(fig)

if __name__ == "__main__":
    # Replace 'model_path' and 'dataset_path' with actual paths to your model and dataset
    evaluate_diffusion_policy_to_video(
        model_path='grid_world/models/best_diffusion_policy.pth',
        dataset_path='grid_world/dataset/grid_world_dataset.pkl',
        output_video_path='grid_world/videos/episode_{}.mp4',
        num_episodes=1
    )
