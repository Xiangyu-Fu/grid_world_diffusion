import torch
import matplotlib.pyplot as plt
from grid_world.common.grid_world_data_loader import PathDataset
from grid_world.common.diffusion_policy import DiffusionPolicy

def evaluate_diffusion_policy(model_path, dataset_path, num_episodes=5):
    # Load the trained model
    policy = DiffusionPolicy()
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    # Load dataset
    delta_timestamps = {
        "observation.state": [-0.1, 0.0],
        "action": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    }
    dataset = PathDataset(dataset_path, delta_timestamps=delta_timestamps)
    device = torch.device("cpu")
    policy.to(device)

    # Evaluate on a few episodes
    for episode in range(num_episodes):
        sample = dataset[episode]
        observation_env = sample['observation.env'].unsqueeze(0).to(device)  # Add batch dimension
        observation_state = sample['observation.state'].unsqueeze(0).to(device)

        # Predict the actions using the trained policy
        predicted_actions = policy.predict(observation_env, observation_state).squeeze(0).cpu().numpy()

        # Plot the environment and the predicted trajectory
        env_map = observation_env.squeeze().cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(env_map, cmap='gray_r')
        plt.scatter(observation_state[0, :, 0].cpu(), observation_state[0, :, 1].cpu(), c='blue', label='Start State')
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
