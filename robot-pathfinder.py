import numpy as np
import pygame
from heapq import heappop, heappush

# initialising pygame
pygame.init()

# setting constants
width, height = 600, 600
grid_size = 10
cell_size = width // grid_size

# grid setup (0 = free, 1 = obstacle)
grid = np.zeros((grid_size, grid_size))
grid[2, 3] = 1  # adding obstacle at (2,3)
grid[5, 0:6] = 1  # adding obstacles from (5,0) to (5,6)

# start and finish positions
start = (0, 0)
finish = (grid_size - 1, grid_size - 1)

# pygame setup
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Robot Pathfinder")
clock = pygame.time.Clock()

# draw grid function
def draw_grid():
    for y in range(grid_size):
        for x in range(grid_size):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            colour = (0, 100, 0) if grid[y, x] == 0 else (255, 0, 0)  # dark green / red
            pygame.draw.rect(screen, colour, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)  # black border

# draw robot function
def draw_robot(pos):
    centre = (pos[0] * cell_size + cell_size // 2, pos[1] * cell_size + cell_size // 2)
    pygame.draw.circle(screen, (0, 0, 255), centre, cell_size // 3)  # blue robot

# draw path function
def draw_path(path):
    if len(path) < 2:
        return
    points = [(x * cell_size + cell_size // 2, y * cell_size + cell_size // 2) for (x, y) in path]
    pygame.draw.lines(screen, (255, 255, 0), False, points, 3)  # yellow path

# a* pathfinding algorithm
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # manhattan distance

def a_star(grid, start, goal):
    neighbours = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-directional movement
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    open_set = []
    heappush(open_set, (fscore[start], start))
    
    while open_set:
        current = heappop(open_set)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        close_set.add(current)
        for dx, dy in neighbours:
            neighbour = current[0] + dx, current[1] + dy
            if 0 <= neighbour[0] < grid_size and 0 <= neighbour[1] < grid_size:
                if grid[neighbour[1]][neighbour[0]] == 1:
                    continue  # skip obstacles
                tentative_g = gscore[current] + 1
                if neighbour in close_set and tentative_g >= gscore.get(neighbour, 0):
                    continue
                if tentative_g < gscore.get(neighbour, 0) or neighbour not in [i[1] for i in open_set]:
                    came_from[neighbour] = current
                    gscore[neighbour] = tentative_g
                    fscore[neighbour] = tentative_g + heuristic(neighbour, goal)
                    heappush(open_set, (fscore[neighbour], neighbour))
    return []  # no path found

# Initialize robot position and path
robot_pos = list(start)
path = a_star(grid, start, finish)
path_index = 0
frame_count = 0

# main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move robot every 30 frames (0.5 seconds at 60 FPS)
    frame_count += 1
    if frame_count % 30 == 0 and path and path_index < len(path):
        robot_pos = list(path[path_index])
        path_index += 1

    # Drawing
    screen.fill((255, 255, 255))  # white background
    draw_grid()
    draw_path(path)
    draw_robot(robot_pos)  # Draw robot at current position
    pygame.display.flip()
    clock.tick(60)

pygame.quit()