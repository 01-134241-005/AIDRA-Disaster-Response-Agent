# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:12:46 2026

@author: Administrator
"""

def dfs(grid, start, goal):
    stack = [(start, [])]
    visited = set()
    rows, cols = len(grid), len(grid[0])
    while stack:
        (x, y), path = stack.pop()
        if (x, y) == goal:
            return path + [(x, y)]
        if (x, y) in visited:
            continue
        visited.add((x, y))
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 'X':
                stack.append(((nx, ny), path + [(x, y)]))
    return None