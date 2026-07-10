"""
Frontier-based exploration — detects boundaries between known and unknown space.

Two types of frontiers:
    1. Exploration frontiers: free ↔ unknown (never observed)
    2. Decay frontiers: certain ↔ decayed (was observed, now uncertain)

Frontiers are clustered into contiguous regions for goal selection.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import ndimage
from slam.occupancy_grid import DecayingOccupancyGrid
from config.settings import NavigationConfig


class Frontier:
    """A cluster of frontier cells."""

    def __init__(self, cells: List[Tuple[int, int]], frontier_type: str = "exploration"):
        """
        Args:
            cells: List of (row, col) grid indices.
            frontier_type: "exploration" or "decay".
        """
        self.cells = cells
        self.frontier_type = frontier_type

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def centroid_grid(self) -> Tuple[float, float]:
        """Centroid in grid coordinates."""
        rows = [c[0] for c in self.cells]
        cols = [c[1] for c in self.cells]
        return (np.mean(rows), np.mean(cols))

    def centroid_world(self, grid: DecayingOccupancyGrid) -> Tuple[float, float]:
        """Centroid in world coordinates."""
        r, c = self.centroid_grid
        return grid.grid_to_world(int(r), int(c))

    def __repr__(self) -> str:
        return f"Frontier({self.frontier_type}, size={self.size})"


class FrontierExplorer:
    """
    Detects and clusters frontiers in the occupancy grid.

    A frontier cell is a FREE cell adjacent to at least one UNKNOWN
    (or DECAYED) cell. Frontiers are clustered using connected-component
    labeling and filtered by minimum size.
    """

    def __init__(self, config: NavigationConfig):
        self.config = config
        self._neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    def detect_frontiers(self, grid: DecayingOccupancyGrid,
                         reexploration_mask: Optional[np.ndarray] = None
                         ) -> List[Frontier]:
        """
        Detect all frontiers in the occupancy grid.

        Args:
            grid: Current occupancy grid.
            reexploration_mask: Optional boolean mask of cells needing
                               re-exploration (from MemoryManager).

        Returns:
            List of Frontier objects, sorted by size (largest first).
        """
        prob_map = grid.get_probability_map()
        shape = grid.shape

        # Define regions
        free_mask = (prob_map < 0.4) & (grid.last_observed >= 0)
        unknown_mask = grid.last_observed < 0  # Never observed
        occupied_mask = prob_map > 0.6

        frontiers = []

        # --- Exploration frontiers (free ↔ unknown) ---
        exploration_frontier_mask = self._find_frontier_cells(
            free_mask, unknown_mask, shape
        )
        exploration_clusters = self._cluster_frontiers(
            exploration_frontier_mask, "exploration"
        )
        frontiers.extend(exploration_clusters)

        # --- Decay frontiers (free ↔ decayed) ---
        if self.config.decay_frontier_enabled and reexploration_mask is not None:
            decay_frontier_mask = self._find_frontier_cells(
                free_mask, reexploration_mask, shape
            )
            # Remove overlap with exploration frontiers
            decay_frontier_mask &= ~exploration_frontier_mask
            decay_clusters = self._cluster_frontiers(
                decay_frontier_mask, "decay"
            )
            frontiers.extend(decay_clusters)

        # Sort by size (largest first)
        frontiers.sort(key=lambda f: f.size, reverse=True)

        return frontiers

    def _find_frontier_cells(self, free_mask: np.ndarray,
                              target_mask: np.ndarray,
                              shape: Tuple[int, int]) -> np.ndarray:
        """
        Find free cells adjacent to target cells.

        A cell is a frontier cell if:
            - It is free (free_mask is True)
            - At least one 8-neighbor is a target cell (target_mask is True)
        """
        frontier = np.zeros(shape, dtype=bool)

        # Use convolution for efficiency: convolve target_mask with a 3x3 kernel
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.float32)

        # Count adjacent target cells
        neighbor_count = ndimage.convolve(
            target_mask.astype(np.float32), kernel,
            mode='constant', cval=0.0
        )

        # Frontier = free AND has at least one target neighbor
        frontier = free_mask & (neighbor_count > 0)

        return frontier

    def _cluster_frontiers(self, frontier_mask: np.ndarray,
                            frontier_type: str) -> List[Frontier]:
        """
        Cluster frontier cells into connected components.

        Args:
            frontier_mask: Boolean mask of frontier cells.
            frontier_type: "exploration" or "decay".

        Returns:
            List of Frontier objects (filtered by min size).
        """
        if not np.any(frontier_mask):
            return []

        # Connected component labeling (8-connectivity)
        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
        labeled, num_features = ndimage.label(frontier_mask, structure=structure)

        frontiers = []
        for label_id in range(1, num_features + 1):
            cells = list(zip(*np.where(labeled == label_id)))
            if len(cells) >= self.config.min_frontier_size:
                frontiers.append(Frontier(cells, frontier_type))

        return frontiers

    def get_nearest_frontier(self, grid: DecayingOccupancyGrid,
                              robot_pos: Tuple[float, float],
                              frontiers: List[Frontier]
                              ) -> Optional[Frontier]:
        """
        Return the nearest frontier to the robot.

        Args:
            grid: Occupancy grid (for coordinate conversion).
            robot_pos: Robot position (x, y) in world coordinates.
            frontiers: List of detected frontiers.

        Returns:
            Nearest Frontier, or None if no frontiers exist.
        """
        if not frontiers:
            return None

        best = None
        best_dist = float('inf')

        for frontier in frontiers:
            cx, cy = frontier.centroid_world(grid)
            dist = np.sqrt((cx - robot_pos[0])**2 + (cy - robot_pos[1])**2)
            if dist < best_dist:
                best_dist = dist
                best = frontier

        return best

    def __repr__(self) -> str:
        return (f"FrontierExplorer(min_size={self.config.min_frontier_size}, "
                f"decay_frontiers={self.config.decay_frontier_enabled})")
