import torch
import matplotlib.pyplot as plt
from grid_world.common.grid_world_data_loader import PathDataset
from grid_world.common.diffusion_policy import DiffusionPolicy
import numpy as np

def evaluate_diffusion_policy(model_path, dataset_path, num_episodes=1):
    # Load the trained model
    policy = DiffusionPolicy()
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    device = torch.device("cpu")

    # load the dataset
    delta_timestamps = {
        "observation.env": [0.0],
        "observation.state": [0.0],
        "action": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,],
    }
    dataset = PathDataset(dataset_path, delta_timestamps=delta_timestamps)

    for episode in range(num_episodes):
        policy.reset()
        index = np.random.randint(0, len(dataset))
        print(f"Episode {episode + 1} - Sample Index: {index}")
        sample = dataset[index]
        observation_env = sample['observation.env'].unsqueeze(0).to(device)  # Add batch dimension
        observation_state = sample['observation.state'].unsqueeze(0).to(device)

        with torch.inference_mode():
            predicted_actions = policy.select_action(sample)
            print(f"Predicted Actions are: {predicted_actions}")

        # Plot the environment and the predicted trajectory
        batch_index = 0 
        time_step = 0      

        env_map = observation_env[batch_index, time_step]
        env_map = env_map.permute(1, 2, 0).numpy()

        plt.figure(0)
        plt.imshow(env_map)
        plt.scatter(observation_state[0, :, 0].cpu(), observation_state[0, :, 1].cpu(), c='green', label='current State')
        plt.plot(predicted_actions[:, 0], predicted_actions[:, 1], c='red', label='Predicted Trajectory')
        plt.title(f'Grid World Play - Episode {episode + 1}')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Replace 'model_path' and 'dataset_path' with actual paths to your model and dataset
    evaluate_diffusion_policy(
        model_path='grid_world/models/diffusion_policy.pth',
        dataset_path='grid_world/dataset/grid_world_dataset.pkl',
        num_episodes=5
    )
