# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:12:10 2026

@author: Administrator
"""

import heapq

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal, hazard_penalty=3):
    rows, cols = len(grid), len(grid[0])
    pq = [(0, start, [])]
    visited = set()

    while pq:
        cost, (x, y), path = heapq.heappop(pq)
        if (x, y) == goal:
            return path + [(x, y)]
        if (x, y) in visited:
            continue
        visited.add((x, y))
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] != 'X':
                    step_cost = 1
                    if grid[nx][ny] == 'R':
                        step_cost += hazard_penalty
                    new_g = len(path) + 1 + step_cost
                    new_f = new_g + heuristic((nx, ny), goal)
                    heapq.heappush(pq, (new_f, (nx, ny), path + [(x, y)]))
    return None