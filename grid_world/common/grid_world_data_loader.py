import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

class PathDataset(Dataset):
    def __init__(self, data_path, delta_timestamps):
        with open(data_path, 'rb') as f:
            self.paths = pickle.load(f)
        
        self.delta_timestamps = delta_timestamps
        
        self.samples = []
        #TODO: add new functions which includes the obstacle information
        for path_index, path in enumerate(self.paths):
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
        Retrurns:
            observation.env: [start_position[x, y], goal_position[x, y], obstacle_position[x, y]xN ]
            observation.state: [x, y]
            action: [[x, y]xlen(delta_timestamps)]

        '''
        path_index, start_time = self.samples[idx]
        path = self.paths[path_index]
        timestamps = path[:, 0]
        
        data = {}

        # TODO: data['observation.env']
        data['observation.env'] = path[0, 1:]

        # data['observation.state']
        state_times = np.array(self.delta_timestamps.get("observation.state", []))
        state_times += start_time
        state_indices = ((state_times - timestamps[0]) / 0.1).astype(int)
        valid_indices = (state_indices >= 0) & (state_indices < len(path))
        state_indices = state_indices[valid_indices]
        data['observation.state'] = path[state_indices, 1:]  # (num_states, 2)
        
        # data['action']
        action_times = np.array(self.delta_timestamps.get("action", []))
        action_times += start_time
        action_indices = ((action_times - timestamps[0]) / 0.1).astype(int)
        actions = []
        for idx in action_indices:
            if idx + 1 < len(path):
                current_state = path[idx, 1:]
                next_state = path[idx + 1, 1:]
                action = next_state - current_state
            else:
                # 如果超出路径长度，用零填充
                action = np.array([0.0, 0.0])
            actions.append(action)
        data['action'] = np.stack(actions)  # (num_actions, 2)
        
        data['observation.image'] = torch.tensor(data['observation.state'], dtype=torch.float32)  # TODO: remove this line to change model backbone
        data['observation.state'] = torch.tensor(data['observation.state'], dtype=torch.float32)
        data['action'] = torch.tensor(data['action'], dtype=torch.float32)
        
        
        return data

