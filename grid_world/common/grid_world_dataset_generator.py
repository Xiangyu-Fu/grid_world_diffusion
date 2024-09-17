import numpy as np
import heapq
import random
import pickle

# Define the size of the grid
world_size = 100

# Create the grid world
grid_world = np.zeros((world_size, world_size))

# Define the position of the obstacle
obstacle_top_left = (40, 40)
obstacle_bottom_right = (60, 60)

# Set the obstacle
grid_world[obstacle_top_left[0]:obstacle_bottom_right[0],
           obstacle_top_left[1]:obstacle_bottom_right[1]] = 1

# Heuristic function
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* algorithm with randomness
def astar_with_randomness(grid, start, goal, randomness=0.3):
    neighbors = [(0,1),(1,0),(0,-1),(-1,0)]

    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        close_set.add(current)
        x, y = current
        
        random.shuffle(neighbors)
        
        for i, j in neighbors:
            neighbor = x + i, y + j
            tentative_g_score = gscore[current] + 1
            
            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue

            if neighbor in close_set:
                continue

            if random.random() < randomness:
                tentative_g_score += random.uniform(0, 2)

            if tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    
    return None

# Get a random point
def get_random_point(grid):
    while True:
        point = (np.random.randint(0, grid.shape[0]), np.random.randint(0, grid.shape[1]))
        if grid[point[0], point[1]] == 0:
            return point

# Generate 5000 paths
num_paths = 5000
paths_data = []

for i in range(num_paths):
    start_point = get_random_point(grid_world)
    end_point = get_random_point(grid_world)
    path = astar_with_randomness(grid_world, start_point, end_point, randomness=0.3)
    
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

# Save the data as a .pkl file
with open('simulation/dataset/grid_world_dataset.pkl', 'wb') as f:
    pickle.dump(paths_data, f)

print("Data has been saved as dataset.pkl")
