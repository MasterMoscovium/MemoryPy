"""
Goal selector — utility-based frontier selection for autonomous exploration.

Evaluates each frontier using a weighted utility function:
    utility = w_info * information_gain
            + w_decay * decay_urgency
            - w_dist * travel_distance

Selects the highest-utility frontier as the next navigation goal.
"""

import numpy as np
from typing import Tuple, Optional, List
from navigation.frontier_explorer import Frontier
from navigation.path_planner import AStarPlanner
from slam.occupancy_grid import DecayingOccupancyGrid
from config.settings import NavigationConfig, NavigationStrategy


class GoalSelector:
    """
    Selects the best frontier goal based on a multi-factor utility function.

    Supports different navigation strategies by adjusting weights:
        - EXPLORATION_HEAVY: Favor large unexplored frontiers
        - BALANCED: Equal consideration of all factors
        - DECAY_PRIORITY: Prioritize re-exploring decayed regions
        - GREEDY: Always go to nearest frontier
        - RANDOM: Pick random frontier
    """

    def __init__(self, config: NavigationConfig,
                 planner: AStarPlanner,
                 rng: Optional[np.random.Generator] = None):
        self.config = config
        self.planner = planner
        self.rng = rng or np.random.default_rng()

        # Get strategy weights
        self.w_info, self.w_decay, self.w_dist = config.get_strategy_weights()

        # Track current goal
        self.current_goal: Optional[Tuple[float, float]] = None
        self.current_path: Optional[List[Tuple[float, float]]] = None
        self._goals_selected = 0

    def select_goal(self, grid: DecayingOccupancyGrid,
                    robot_pos: Tuple[float, float],
                    frontiers: List[Frontier],
                    reexploration_mask: Optional[np.ndarray] = None
                    ) -> Optional[Tuple[float, float]]:
        """
        Select the best frontier goal.

        Args:
            grid: Current occupancy grid.
            robot_pos: Robot position (x, y) in world coordinates.
            frontiers: Detected frontiers from FrontierExplorer.
            reexploration_mask: Boolean mask of decayed cells.

        Returns:
            Goal position (x, y) in world coordinates, or None if no
            valid frontiers exist.
        """
        if not frontiers:
            return None

        # Random strategy
        if self.config.strategy == NavigationStrategy.RANDOM:
            frontier = frontiers[self.rng.integers(0, len(frontiers))]
            goal = frontier.centroid_world(grid)
            self.current_goal = goal
            self._goals_selected += 1
            return goal

        # Evaluate utility for each frontier
        best_utility = -float('inf')
        best_goal = None
        best_path = None

        for frontier in frontiers:
            goal = frontier.centroid_world(grid)

            # --- Information gain ---
            info_gain = self._compute_info_gain(frontier, grid)

            # --- Decay urgency ---
            decay_urgency = self._compute_decay_urgency(
                frontier, grid, reexploration_mask
            )

            # --- Travel distance (normalized) ---
            distance = np.sqrt((goal[0] - robot_pos[0])**2 +
                               (goal[1] - robot_pos[1])**2)
            # Normalize distance by environment diagonal
            diag = np.sqrt(grid.grid_config.width_m**2 +
                           grid.grid_config.height_m**2)
            norm_distance = distance / max(diag, 1.0)

            # --- Compute utility ---
            utility = (self.w_info * info_gain +
                       self.w_decay * decay_urgency -
                       self.w_dist * norm_distance)

            if utility > best_utility:
                best_utility = utility
                best_goal = goal

        if best_goal is not None:
            self.current_goal = best_goal
            self._goals_selected += 1

            # Try to plan a path
            path = self.planner.plan(grid, robot_pos, best_goal)
            self.current_path = path

        return best_goal

    def _compute_info_gain(self, frontier: Frontier,
                            grid: DecayingOccupancyGrid) -> float:
        """
        Estimate information gain from visiting a frontier.

        Based on frontier size (more cells = more potential information)
        and surrounding unknown area.
        """
        # Base: frontier size (normalized)
        max_size = max(grid.shape[0] * grid.shape[1] * 0.01, 1.0)
        size_score = frontier.size / max_size

        # Bonus for exploration frontiers (higher priority than decay)
        type_bonus = 1.0 if frontier.frontier_type == "exploration" else 0.5

        return size_score * type_bonus

    def _compute_decay_urgency(self, frontier: Frontier,
                                grid: DecayingOccupancyGrid,
                                reexploration_mask: Optional[np.ndarray]
                                ) -> float:
        """
        Compute decay urgency near this frontier.

        Higher when many high-value cells near the frontier
        are close to the decay threshold.
        """
        if reexploration_mask is None or not np.any(reexploration_mask):
            return 0.0

        if frontier.frontier_type != "decay":
            return 0.0

        # Count re-exploration cells in vicinity of frontier
        total_reexplore = 0
        search_radius = 5  # cells

        for r, c in frontier.cells[:50]:  # Cap for performance
            r_min = max(0, r - search_radius)
            r_max = min(grid.shape[0], r + search_radius + 1)
            c_min = max(0, c - search_radius)
            c_max = min(grid.shape[1], c + search_radius + 1)

            total_reexplore += np.sum(reexploration_mask[r_min:r_max, c_min:c_max])

        # Normalize
        max_reexplore = max(np.sum(reexploration_mask), 1)
        return total_reexplore / max_reexplore

    # ------------------------------------------------------------------
    # Path following
    # ------------------------------------------------------------------

    def get_next_waypoint(self, robot_pos: Tuple[float, float],
                          lookahead: float = 0.5
                          ) -> Optional[Tuple[float, float]]:
        """
        Get the next waypoint along the current path.

        Uses a simple lookahead: returns the first path point
        that is at least `lookahead` meters ahead of the robot.

        Args:
            robot_pos: Current robot position (x, y).
            lookahead: Minimum distance ahead to look.

        Returns:
            Next waypoint (x, y), or None if no path.
        """
        if self.current_path is None or len(self.current_path) == 0:
            return self.current_goal

        # Find the first waypoint beyond lookahead distance
        for wp in self.current_path:
            dist = np.sqrt((wp[0] - robot_pos[0])**2 +
                           (wp[1] - robot_pos[1])**2)
            if dist >= lookahead:
                return wp

        # If all waypoints are close, return the last one (goal)
        return self.current_path[-1]

    def is_goal_reached(self, robot_pos: Tuple[float, float],
                         threshold: float = 0.5) -> bool:
        """Check if the robot is close enough to the current goal."""
        if self.current_goal is None:
            return True
        dist = np.sqrt((robot_pos[0] - self.current_goal[0])**2 +
                       (robot_pos[1] - self.current_goal[1])**2)
        return dist < threshold

    def should_replan(self, timestep: int) -> bool:
        """Check if we should replan the path."""
        return timestep % self.config.replan_interval == 0

    def clear_goal(self):
        """Clear the current goal and path."""
        self.current_goal = None
        self.current_path = None

    @property
    def goals_selected(self) -> int:
        return self._goals_selected

    def __repr__(self) -> str:
        return (f"GoalSelector(strategy={self.config.strategy.value}, "
                f"w=[{self.w_info}, {self.w_decay}, {self.w_dist}], "
                f"goals={self._goals_selected})")
