import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt

class PathDataset(Dataset):
    def __init__(self, data_path, delta_timestamps, ):
        with open(data_path, 'rb') as f:
            self.paths = pickle.load(f)
        
        self.delta_timestamps = delta_timestamps
        self.max_observation_length = len(delta_timestamps.get("observation.env", []))
        self.max_state_length = len(delta_timestamps.get("observation.state", []))
        self.max_action_length = len(delta_timestamps.get("action", []))
        
        self.samples = []
        for path_index, path_data in enumerate(self.paths):
            path = np.array(path_data['path'])
            timestamps = path[:, 0]
            max_time = timestamps[-1]

            max_start_time = max_time - max(self.delta_timestamps["action"])
            num_samples = int((max_start_time - timestamps[0]) / 0.1) + 1
            for i in range(num_samples):
                start_time = timestamps[0] + i * 0.1
                self.samples.append((path_index, start_time))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        '''
        Returns:
            observation.env: 2D binary map with obstacles, start, and goal
            observation.state: [x, y]
            action: [[x, y]*len(delta_timestamps)]
        '''
        path_index, start_time = self.samples[idx]
        path_data = self.paths[path_index]
        path = np.array(path_data['path'])
        obstacle_points = np.array(path_data['obstacle'])
        timestamps = path[:, 0]
        
        data = {}
        # data['observation.env'] - Create a 2D binary map
        grid_size = 100  # Assuming the world size is 100x100
        env_map = np.zeros((3, grid_size, grid_size), dtype=np.float32)
        

        for i in range(3):        
            for point in obstacle_points:
                x, y = point
                env_map[i, x, y] = 1.0
            
            start_position = path[0, 1:].astype(int)
            goal_position = path[-1, 1:].astype(int)
            env_map[i, start_position[0], start_position[1]] = 0.5  # Start position
            env_map[i, goal_position[0], goal_position[1]] = 0.75  # Goal position
            
        data['observation.env'] = env_map  # (channel, grid_size, grid_size)

        # data['observation.state']
        state_times = np.array(self.delta_timestamps.get("observation.state", []))
        state_times += start_time
        state_indices = ((state_times - timestamps[0]) / 0.1).astype(int)
        valid_indices = (state_indices >= 0) & (state_indices < len(path))
        state_indices = state_indices[valid_indices]
        observation_state = path[state_indices, 1:]  # (num_states, 2)

        # Pad observation.state to max_state_length
        if len(observation_state) < self.max_state_length:
            pad_length = self.max_state_length - len(observation_state)
            observation_state = np.pad(observation_state, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
        else:
            observation_state = observation_state[:self.max_state_length]
        data['observation.state'] = observation_state
        
        # data['action']
        action_times = np.array(self.delta_timestamps.get("action", []))
        action_times += start_time
        action_indices = ((action_times - timestamps[0]) / 0.1).astype(int)
        action = path[action_indices, 1:]  # (num_actions, 2)

        # Pad action to max_action_length
        if len(action) < self.max_action_length:
            pad_length = self.max_action_length - len(action)
            action = np.pad(action, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
        else:
            action = action[:self.max_action_length]
        data['action'] = action

        data['observation.env'] = torch.tensor(data['observation.env'], dtype=torch.float32).unsqueeze(0).repeat(self.max_observation_length, 1, 1, 1)  # Add channel dimension
        data['observation.state'] = torch.tensor(data['observation.state'], dtype=torch.float32)
        data['action'] = torch.tensor(data['action'], dtype=torch.float32)
        
        return data
    
    def sample_env(self):
        """
        Returns:
            observation.env: 2D binary map with obstacles, start, and goal
            observation.state: [x, y] at a random time
        """
        # 随机选择一个路径索引
        path_index = np.random.randint(0, len(self.paths))
        path_data = self.paths[path_index]
        path = np.array(path_data['path'])
        obstacle_points = np.array(path_data['obstacle'])
        timestamps = path[:, 0]
        
        # 随机选择一个时间点
        random_time = np.random.uniform(timestamps[0], timestamps[-1])
        
        data = {}

        # data['observation.env'] - Create a 2D binary map
        grid_size = 100  # 假设世界的大小为100x100
        env_map = np.zeros((3, grid_size, grid_size), dtype=np.float32)
        
        for i in range(3):        
            for point in obstacle_points:
                x, y = point
                env_map[i, x, y] = 1.0
            
            start_position = path[0, 1:].astype(int)
            goal_position = path[-1, 1:].astype(int)
            env_map[i, start_position[0], start_position[1]] = 0.5  # Start position
            env_map[i, goal_position[0], goal_position[1]] = 0.75  # Goal position
            
        data['observation.env'] = env_map  # (channel, grid_size, grid_size)

        # # plot the environment
        # plt.figure()
        # plt.imshow(np.transpose(env_map, (1, 2, 0)))  
        # plt.title('Environment')
        # plt.show()


        # data['observation.state'] - Get state at random time
        state_index = int((random_time - timestamps[0]) / 0.1)
        state_index = min(max(state_index, 0), len(path) - 1)  # 保证索引在合法范围内
        observation_state = path[state_index, 1:]  # [x, y] 位置

        data['observation.state'] = observation_state
        
        # 将 numpy 转换为 tensor
        data['observation.env'] = torch.tensor(data['observation.env'], dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        data['observation.state'] = torch.tensor(data['observation.state'], dtype=torch.float32).unsqueeze(0)

        return data

