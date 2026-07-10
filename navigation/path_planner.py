"""
A* path planner with decay-aware cost function.

Plans paths on the occupancy grid with costs that account for
occupancy probability, uncertainty, and unknown regions.
Supports replanning when decay invalidates the current path.
"""

import numpy as np
import heapq
from typing import Tuple, List, Optional
from slam.occupancy_grid import DecayingOccupancyGrid
from config.settings import NavigationConfig


class AStarPlanner:
    """
    A* path planner on the occupancy grid.

    Cost function:
        cost(cell) = base_cost + w_occ * P(occupied) + w_unc * uncertainty
    
    Unknown/decayed cells have elevated cost (conservative planning).
    """

    def __init__(self, config: NavigationConfig):
        self.config = config
        # 8-connected neighbors with movement costs
        self._neighbors = [
            (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
            (0, -1, 1.0),                   (0, 1, 1.0),
            (1, -1, 1.414),  (1, 0, 1.0),   (1, 1, 1.414),
        ]

    def plan(self, grid: DecayingOccupancyGrid,
             start_world: Tuple[float, float],
             goal_world: Tuple[float, float],
             max_iterations: int = 50000
             ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan a path from start to goal in world coordinates.

        Args:
            grid: Current occupancy grid.
            start_world: Start position (x, y) in meters.
            goal_world: Goal position (x, y) in meters.
            max_iterations: Maximum A* iterations (prevents runaway).

        Returns:
            List of (x, y) waypoints in world coordinates,
            or None if no path found.
        """
        # Convert to grid coordinates
        start_rc = grid.world_to_grid(start_world[0], start_world[1])
        goal_rc = grid.world_to_grid(goal_world[0], goal_world[1])

        # Validate start and goal
        if not grid._in_bounds(start_rc[0], start_rc[1]):
            return None
        if not grid._in_bounds(goal_rc[0], goal_rc[1]):
            return None

        # Precompute cost map
        cost_map = self._build_cost_map(grid)

        # Check if goal is reachable (not in a wall)
        if cost_map[goal_rc[0], goal_rc[1]] > self.config.obstacle_cost * 0.5:
            # Goal is in obstacle — find nearest free cell
            goal_rc = self._nearest_free_cell(cost_map, goal_rc, grid.shape)
            if goal_rc is None:
                return None

        # A* search
        path_grid = self._astar(start_rc, goal_rc, cost_map, grid.shape,
                                max_iterations)

        if path_grid is None:
            return None

        # Convert path to world coordinates
        path_world = []
        for r, c in path_grid:
            wx, wy = grid.grid_to_world(r, c)
            path_world.append((wx, wy))

        return path_world

    def _build_cost_map(self, grid: DecayingOccupancyGrid) -> np.ndarray:
        """
        Build the traversal cost map from the occupancy grid.

        Costs:
            - Free cells: base_cost
            - Uncertain cells: base_cost + w_uncertainty * uncertainty
            - Occupied cells: very high cost (effectively blocked)
            - Unknown cells: moderate cost (conservative)
        """
        prob_map = grid.get_probability_map()
        uncertainty = grid.get_uncertainty_map()
        cfg = self.config

        cost_map = np.full(grid.shape, cfg.base_cost, dtype=np.float32)

        # Add occupancy cost
        cost_map += cfg.obstacle_cost * prob_map

        # Add uncertainty cost
        cost_map += cfg.uncertainty_cost * uncertainty

        # Heavily penalize definitely-occupied cells
        occupied = prob_map > 0.7
        cost_map[occupied] = cfg.obstacle_cost * 10

        return cost_map

    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int],
               cost_map: np.ndarray, shape: Tuple[int, int],
               max_iterations: int
               ) -> Optional[List[Tuple[int, int]]]:
        """
        Core A* algorithm on grid.

        Returns list of (row, col) from start to goal, or None.
        """
        rows, cols = shape

        # Priority queue: (f_score, counter, row, col)
        counter = 0
        open_set = [(0.0, counter, start[0], start[1])]
        came_from = {}

        g_score = np.full(shape, float('inf'), dtype=np.float64)
        g_score[start[0], start[1]] = 0.0

        closed = np.zeros(shape, dtype=bool)

        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            f, _, cr, cc = heapq.heappop(open_set)

            if (cr, cc) == goal:
                return self._reconstruct_path(came_from, goal)

            if closed[cr, cc]:
                continue
            closed[cr, cc] = True

            for dr, dc, move_cost in self._neighbors:
                nr, nc = cr + dr, cc + dc

                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if closed[nr, nc]:
                    continue

                # Cell traversal cost
                cell_cost = cost_map[nr, nc]
                if cell_cost > self.config.obstacle_cost * 5:
                    continue  # Impassable

                new_g = g_score[cr, cc] + move_cost * cell_cost

                if new_g < g_score[nr, nc]:
                    g_score[nr, nc] = new_g
                    f_score = new_g + self._heuristic(nr, nc, goal[0], goal[1])
                    came_from[(nr, nc)] = (cr, cc)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, nr, nc))

        return None  # No path found

    @staticmethod
    def _heuristic(r1: int, c1: int, r2: int, c2: int) -> float:
        """Euclidean distance heuristic."""
        return np.sqrt((r1 - r2)**2 + (c1 - c2)**2)

    @staticmethod
    def _reconstruct_path(came_from: dict,
                           goal: Tuple[int, int]
                           ) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dict."""
        path = [goal]
        current = goal
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _nearest_free_cell(cost_map: np.ndarray,
                            target: Tuple[int, int],
                            shape: Tuple[int, int],
                            search_radius: int = 20
                            ) -> Optional[Tuple[int, int]]:
        """Find nearest low-cost cell to the target."""
        best = None
        best_dist = float('inf')
        tr, tc = target

        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                nr, nc = tr + dr, tc + dc
                if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                    if cost_map[nr, nc] < 50:
                        dist = abs(dr) + abs(dc)
                        if dist < best_dist:
                            best_dist = dist
                            best = (nr, nc)

        return best

    def get_path_cost(self, path_world: List[Tuple[float, float]]) -> float:
        """Compute total Euclidean length of a world-coordinate path."""
        if not path_world or len(path_world) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(path_world)):
            dx = path_world[i][0] - path_world[i-1][0]
            dy = path_world[i][1] - path_world[i-1][1]
            total += np.sqrt(dx**2 + dy**2)
        return total

    def __repr__(self) -> str:
        return (f"AStarPlanner(base={self.config.base_cost}, "
                f"occ_cost={self.config.obstacle_cost}, "
                f"unc_cost={self.config.uncertainty_cost})")
