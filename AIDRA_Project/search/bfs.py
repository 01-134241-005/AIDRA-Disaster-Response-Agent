# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:12:31 2026

@author: Administrator
"""

from collections import deque

def bfs(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    q = deque([(start, [])])
    visited = set()
    while q:
        (x, y), path = q.popleft()
        if (x, y) == goal:
            return path + [(x, y)]
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] != 'X' and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append(((nx, ny), path + [(x, y)]))
    return None