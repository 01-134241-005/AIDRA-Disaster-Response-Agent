import random
from environment import get_victims

def block_road(grid):
    victim_locations = [loc for loc, _ in get_victims()]
    # Try up to 20 times to find a free, non-victim, non-start, non-hospital cell
    for _ in range(20):
        x = random.randint(0, len(grid)-1)
        y = random.randint(0, len(grid[0])-1)
        # Only block empty cells ('.') that are not a victim location
        if grid[x][y] == '.' and (x, y) not in victim_locations:
            grid[x][y] = 'X'
            return (x, y)
    return None  # No suitable cell to block