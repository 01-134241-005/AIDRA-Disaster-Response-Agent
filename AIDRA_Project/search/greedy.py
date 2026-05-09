# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:13:02 2026

@author: Administrator
"""

import heapq

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def greedy(grid, start, goal):
    pq = [(heuristic(start, goal), start, [])]
    visited = set()
    rows, cols = len(grid), len(grid[0])
    while pq:
        _, (x, y), path = heapq.heappop(pq)
        if (x, y) == goal:
            return path + [(x, y)]
        if (x, y) in visited:
            continue
        visited.add((x, y))
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 'X':
                heapq.heappush(pq, (heuristic((nx, ny), goal), (nx, ny), path + [(x, y)]))
    return None