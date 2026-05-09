# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:13:10 2026

@author: Administrator
"""

def hill_climbing(grid, start, goal):
    current = start
    path = [current]
    visited = set()
    rows, cols = len(grid), len(grid[0])
    while current != goal:
        visited.add(current)
        x, y = current
        neighbors = []
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] != 'X' and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
        if not neighbors:
            return None
        # choose neighbor closest to goal
        next_node = min(neighbors, key=lambda n: abs(n[0]-goal[0]) + abs(n[1]-goal[1]))
        current = next_node
        path.append(current)
    return path