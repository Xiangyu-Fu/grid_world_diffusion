import numpy as np
import heapq
import random
import pickle
import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap

# Define grid size
world_size = 100

# Create grid world
grid_world = np.zeros((world_size, world_size))

# Define obstacle positions
obstacle_top_left = (40, 40)
obstacle_bottom_right = (60, 60)

# Set obstacles
grid_world[obstacle_top_left[0]:obstacle_bottom_right[0],
           obstacle_top_left[1]:obstacle_bottom_right[1]] = 1

# Heuristic function (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Get a random point not within obstacles
def get_random_point(grid):
    while True:
        point = (np.random.randint(0, grid.shape[0]), np.random.randint(0, grid.shape[1]))
        if grid[point[0], point[1]] == 0:
            return point

# Modified path generation function
def generate_path_with_random_walks(grid, start, goal):
    current_position = start
    path = [current_position]
    total_steps = 0
    max_attempts = 10000  # Prevent infinite loop
    steps_since_last_random_walk = 0
    next_random_walk_interval = random.randint(5, 15)  # Initial random interval

    while current_position != goal and total_steps < max_attempts:
        steps_since_last_random_walk += 1

        # Check if random walk is needed
        if steps_since_last_random_walk >= next_random_walk_interval:
            # Random walk
            random_walk_steps = random.randint(1, 10)
            for _ in range(random_walk_steps):
                neighbors = [ (current_position[0] + dx, current_position[1] + dy)
                              for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)] ]
                valid_neighbors = [ neighbor for neighbor in neighbors
                                    if 0 <= neighbor[0] < grid.shape[0]
                                    and 0 <= neighbor[1] < grid.shape[1]
                                    and grid[neighbor[0], neighbor[1]] == 0
                                    and neighbor not in path ]
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
            neighbors = [ (current_position[0] + dx, current_position[1] + dy)
                          for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)] ]
            valid_neighbors = [ neighbor for neighbor in neighbors
                                if 0 <= neighbor[0] < grid.shape[0]
                                and 0 <= neighbor[1] < grid.shape[1]
                                and grid[neighbor[0], neighbor[1]] == 0
                                and neighbor not in path ]
            if not valid_neighbors:
                break  # No valid path, exit
            # Choose neighbor with lowest heuristic cost
            min_neighbor = min(valid_neighbors, key=lambda x: heuristic(x, goal))
            current_position = min_neighbor
            path.append(current_position)
            total_steps += 1
            if current_position == goal:
                break

    if current_position != goal:
        return None  # Path not found

    return path

# Generate 5000 paths and save as .pkl file
num_paths = 5000
paths_data = []

for i in range(num_paths):
    start_point = get_random_point(grid_world)
    end_point = get_random_point(grid_world)
    path = generate_path_with_random_walks(grid_world, start_point, end_point)

    if path is None or len(path) < 2:
        continue

    path_points = []
    timestamp = 0.0
    for point in path:
        x, y = point
        path_points.append([timestamp, x, y])
        timestamp += 0.1

    paths_data.append(np.array(path_points))

    print(f"\r {i+1}/{num_paths} paths generated", end='')

# Save data as .pkl file
with open('grid_world/dataset/grid_world_dataset.pkl', 'wb') as f:
    pickle.dump(paths_data, f)

print("\nData has been saved as grid_world_dataset.pkl")

# Visualize 10 paths on the same figure
import matplotlib.pyplot as plt

# If the number of paths is less than 10, take all
num_visualize = min(10, len(paths_data))

plt.figure(figsize=(8,8))
plt.imshow(grid_world, cmap='gray_r')

# Use get_cmap from matplotlib.cm or plt to get color map
try:
    colors = plt.get_cmap('tab10')
except AttributeError:
    from matplotlib.cm import get_cmap
    colors = get_cmap('tab10')

for i in range(num_visualize):
    path = paths_data[i]
    x_vals = path[:,1]
    y_vals = path[:,2]

    plt.plot(y_vals, x_vals, color=colors(i % 10), linewidth=2, label=f'Path {i+1}')
    plt.scatter(y_vals[0], x_vals[0], color=colors(i % 10), marker='o')  # Start point
    plt.scatter(y_vals[-1], x_vals[-1], color=colors(i % 10), marker='x')   # End point

plt.title('Visualization of 10 Paths on One Figure')
plt.legend()
plt.grid(True)
plt.show()
