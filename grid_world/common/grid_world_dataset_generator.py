import numpy as np
import heapq
import random
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

class GridWorldPathGenerator:
    def __init__(self, world_size=100, num_paths=5000, min_distance=10):
        self.world_size = world_size
        self.num_paths = num_paths
        self.min_distance = min_distance
        self.paths_data = []

    # Heuristic function (Manhattan distance)
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Get a random point not within obstacles
    def get_random_point(self, grid):
        while True:
            point = (np.random.randint(0, grid.shape[0]), np.random.randint(0, grid.shape[1]))
            if grid[point[0], point[1]] == 0:
                return point

    # Get two random points that are not too close to each other
    def get_two_distant_points(self, grid):
        while True:
            start = self.get_random_point(grid)
            goal = self.get_random_point(grid)
            if self.heuristic(start, goal) >= self.min_distance:
                return start, goal

    # Modified path generation function
    def generate_path_with_random_walks(self, grid, start, goal):
        current_position = start
        path = [current_position]
        total_steps = 0
        max_attempts = 50000  # Prevent infinite loop
        steps_since_last_random_walk = 0
        next_random_walk_interval = random.randint(5, 15)  # Initial random interval

        while current_position != goal and total_steps < max_attempts:
            steps_since_last_random_walk += 1

            # Check if random walk is needed
            if steps_since_last_random_walk >= next_random_walk_interval:
                # Random walk
                random_walk_steps = random.randint(1, 1)
                for _ in range(random_walk_steps):
                    neighbors = [(current_position[0] + dx, current_position[1] + dy)
                                 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
                    valid_neighbors = [neighbor for neighbor in neighbors
                                       if 0 <= neighbor[0] < grid.shape[0]
                                       and 0 <= neighbor[1] < grid.shape[1]
                                       and grid[neighbor[0], neighbor[1]] == 0
                                       and neighbor not in path]
                    if not valid_neighbors:
                        break  # No valid path, exit random walk
                    current_position = random.choice(valid_neighbors)
                    path.append(current_position)
                    total_steps += 1
                    if current_position == goal:
                        break
                # Reset counter and next random interval
                steps_since_last_random_walk = 0
                next_random_walk_interval = random.randint(5, 15)
            else:
                # Move one step according to heuristic algorithm
                neighbors = [(current_position[0] + dx, current_position[1] + dy)
                             for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
                valid_neighbors = [neighbor for neighbor in neighbors
                                   if 0 <= neighbor[0] < grid.shape[0]
                                   and 0 <= neighbor[1] < grid.shape[1]
                                   and grid[neighbor[0], neighbor[1]] == 0
                                   and neighbor not in path]
                if not valid_neighbors:
                    break  # No valid path, exit
                # Choose neighbor with lowest heuristic cost
                min_neighbor = min(valid_neighbors, key=lambda x: self.heuristic(x, goal))
                current_position = min_neighbor
                path.append(current_position)
                total_steps += 1
                if current_position == goal:
                    break

        if current_position != goal:
            return None  # Path not found

        return path

    # Generate paths and save as .pkl file
    def generate_paths(self):
        i = 0
        with tqdm(total=self.num_paths, desc="Generating paths") as pbar:
            while i < self.num_paths:
                # Create grid world
                grid_world = np.zeros((self.world_size, self.world_size))

                # Generate 1 to 3 random obstacles
                num_obstacles = random.randint(10, 30)
                obstacle_points = []

                for _ in range(num_obstacles):
                    obstacle_type = random.choice(['rectangle', 'circle'])

                    if obstacle_type == 'rectangle':
                        # Randomly generate top left and bottom right corners for the rectangle
                        top_left_x = random.randint(0, self.world_size - 1)
                        top_left_y = random.randint(0, self.world_size - 1)
                        width = random.randint(5, 20)  # Random width of the rectangle
                        height = random.randint(5, 20)  # Random height of the rectangle

                        bottom_right_x = min(top_left_x + width, self.world_size - 1)
                        bottom_right_y = min(top_left_y + height, self.world_size - 1)

                        # Set the obstacle area to 1 and store obstacle points
                        for x in range(top_left_x, bottom_right_x):
                            for y in range(top_left_y, bottom_right_y):
                                grid_world[x, y] = 1
                                obstacle_points.append([x, y])

                    elif obstacle_type == 'circle':
                        # Randomly generate center and radius for the circle
                        center_x = random.randint(0, self.world_size - 1)
                        center_y = random.randint(0, self.world_size - 1)
                        radius = random.randint(5, 10)  # Random radius of the circle

                        for x in range(self.world_size):
                            for y in range(self.world_size):
                                # Use the equation of a circle to determine if a point is inside
                                if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                                    grid_world[x, y] = 1
                                    obstacle_points.append([x, y])

                start_point, end_point = self.get_two_distant_points(grid_world)
                path = self.generate_path_with_random_walks(grid_world, start_point, end_point)

                if path is None or len(path) < 2:
                    continue

                path_points = []
                timestamp = 0.0
                for point in path:
                    x, y = point
                    path_points.append([timestamp, x, y])
                    timestamp += 0.1

                self.paths_data.append({'path': path_points, 'obstacle': obstacle_points})
                i += 1
                pbar.update(1)

        # Save data as .pkl file
        with open('grid_world/dataset/grid_world_dataset_test.pkl', 'wb') as f:
            pickle.dump(self.paths_data, f)

        print("\nData has been saved as grid_world_dataset.pkl")

    # Randomly display a few paths
    def visualize_paths(self, num_visualize=5):
        print(f"Data contains {len(self.paths_data)} paths")
        num_visualize = min(num_visualize, len(self.paths_data))
        plt.figure(figsize=(8, 8))

        for i in range(num_visualize):
            grid_world = np.zeros((self.world_size, self.world_size))
            obstacle_points = self.paths_data[i]['obstacle']

            for point in obstacle_points:
                grid_world[point[0], point[1]] = 1

            plt.imshow(grid_world, cmap='gray_r')
            path = self.paths_data[i]['path']
            x_vals = [p[1] for p in path]
            y_vals = [p[2] for p in path]
            plt.plot(y_vals, x_vals, linewidth=2, label=f'Path {i + 1}')
            plt.scatter(y_vals[0], x_vals[0], marker='o')  # Start point
            plt.scatter(y_vals[-1], x_vals[-1], marker='x')  # End point
            plt.title(f'Visualization of Path {i + 1}')
            plt.legend()
            plt.grid(True)
            plt.show()

# Example usage
if __name__ == "__main__":
    generator = GridWorldPathGenerator(world_size=100, num_paths=500, min_distance=80)
    generator.generate_paths()
    generator.visualize_paths(num_visualize=5)