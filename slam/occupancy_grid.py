"""
Decaying Occupancy Grid — Log-odds occupancy mapping with memory decay support.

Each cell stores:
    - log_odds: occupancy probability in log-odds form
    - last_observed: timestamp of last sensor observation
    - visit_count: number of times the cell was observed

Decay pulls log-odds toward 0 (maximum uncertainty, P=0.5),
modeling genuine *forgetting* — the robot becomes unsure, not wrong.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from config.settings import GridConfig, LidarConfig


class DecayingOccupancyGrid:
    """
    Occupancy grid with log-odds representation and memory decay hooks.

    Log-odds advantages:
        - Updates are simple additions (not multiplications)
        - Numerically stable for long sequences
        - Easy to clamp to prevent overconfidence

    Coordinate convention:
        - Grid: (row, col), origin top-left
        - World: (x, y) in meters, origin bottom-left
    """

    def __init__(self, grid_config: GridConfig, lidar_config: LidarConfig):
        """
        Initialize an empty occupancy grid (all cells unknown/P=0.5).

        Args:
            grid_config: Grid dimensions and log-odds parameters.
            lidar_config: Sensor model parameters (p_occupied, p_free, etc.).
        """
        self.grid_config = grid_config
        self.lidar_config = lidar_config

        shape = grid_config.shape  # (rows, cols)

        # Core occupancy data (log-odds)
        self.log_odds = np.full(shape, grid_config.log_odds_prior,
                                dtype=np.float32)

        # Memory metadata
        self.last_observed = np.full(shape, -1.0, dtype=np.float32)  # -1 = never
        self.visit_count = np.zeros(shape, dtype=np.int32)

        # Precompute inverse sensor model log-odds
        self._l_occ = np.log(lidar_config.p_occupied /
                             (1.0 - lidar_config.p_occupied))
        self._l_free = np.log(lidar_config.p_free /
                              (1.0 - lidar_config.p_free))
        self._l_prior = grid_config.log_odds_prior

        # Resolution shortcut
        self.resolution = grid_config.resolution

    # ------------------------------------------------------------------
    # Grid update (inverse sensor model)
    # ------------------------------------------------------------------

    def update(self, pose: Tuple[float, float, float],
               scan: np.ndarray, timestamp: float):
        """
        Update the grid using a LiDAR scan and the inverse sensor model.

        For each beam:
            - Cells along the beam path → marked FREE
            - Cells near the hit point → marked OCCUPIED
            - Timestamps and visit counts updated

        Args:
            pose: Robot pose (x, y, θ) in world coordinates.
            scan: Array of shape (N, 2) with [angle, distance] per beam.
            timestamp: Current simulation time (for decay tracking).
            
        Returns:
            Tuple of numpy arrays: (free_r, free_c, occ_r, occ_c) for reuse.
        """
        x, y, theta = pose
        
        free_r_list, free_c_list = [], []
        occ_r_list, occ_c_list = [], []

        for i in range(len(scan)):
            angle = scan[i, 0]
            distance = scan[i, 1]

            if distance <= self.lidar_config.min_range:
                continue

            # Compute hit point
            hit_x = x + distance * np.cos(angle)
            hit_y = y + distance * np.sin(angle)

            # Get cells along the beam using Bresenham
            r0, c0 = self.world_to_grid(x, y)
            r1, c1 = self.world_to_grid(hit_x, hit_y)
            cells = self._bresenham(r0, c0, r1, c1)

            is_hit = distance < self.lidar_config.max_range - 0.05

            for j, (row, col) in enumerate(cells):
                if not self._in_bounds(row, col):
                    continue

                if j == len(cells) - 1 and is_hit:
                    # Last cell = hit point → occupied
                    occ_r_list.append(row)
                    occ_c_list.append(col)
                    self.log_odds[row, col] += self._l_occ - self._l_prior
                    # Also mark neighboring cells as occupied (wall thickness)
                    wt = self.lidar_config.wall_thickness
                    for dr in range(-wt + 1, wt):
                        for dc in range(-wt + 1, wt):
                            nr, nc = row + dr, col + dc
                            if self._in_bounds(nr, nc) and (dr != 0 or dc != 0):
                                self.log_odds[nr, nc] += (
                                    (self._l_occ - self._l_prior) * 0.3
                                )
                else:
                    # Cells before hit → free
                    free_r_list.append(row)
                    free_c_list.append(col)
                    self.log_odds[row, col] += self._l_free - self._l_prior

                # Update metadata
                self.last_observed[row, col] = timestamp
                self.visit_count[row, col] += 1

        # Clamp log-odds to prevent overconfidence
        np.clip(self.log_odds,
                self.grid_config.log_odds_min,
                self.grid_config.log_odds_max,
                out=self.log_odds)
                
        return (np.array(free_r_list, dtype=np.int32), 
                np.array(free_c_list, dtype=np.int32), 
                np.array(occ_r_list, dtype=np.int32), 
                np.array(occ_c_list, dtype=np.int32))

    def update_vectorized(self, free_r: np.ndarray, free_c: np.ndarray,
                          occ_r: np.ndarray, occ_c: np.ndarray,
                          timestamp: float):
        """
        Fast, vectorized update using precomputed indices.
        Significantly faster than calling update() which loops Bresenham per beam.
        """
        # Apply free updates
        if len(free_r) > 0:
            self.log_odds[free_r, free_c] += (self._l_free - self._l_prior)
            self.last_observed[free_r, free_c] = timestamp
            self.visit_count[free_r, free_c] += 1
            
        # Apply occupied updates (with thickness)
        if len(occ_r) > 0:
            self.log_odds[occ_r, occ_c] += (self._l_occ - self._l_prior)
            self.last_observed[occ_r, occ_c] = timestamp
            self.visit_count[occ_r, occ_c] += 1

            # Thickness
            wt = self.lidar_config.wall_thickness
            for dr in range(-wt + 1, wt):
                for dc in range(-wt + 1, wt):
                    if dr == 0 and dc == 0: continue
                    nr, nc = occ_r + dr, occ_c + dc
                    valid = (nr >= 0) & (nr < self.shape[0]) & (nc >= 0) & (nc < self.shape[1])
                    v_nr, v_nc = nr[valid], nc[valid]
                    if len(v_nr) > 0:
                        self.log_odds[v_nr, v_nc] += (self._l_occ - self._l_prior) * 0.3
                        self.last_observed[v_nr, v_nc] = timestamp

        # Clamp
        np.clip(self.log_odds,
                self.grid_config.log_odds_min,
                self.grid_config.log_odds_max,
                out=self.log_odds)

    # ------------------------------------------------------------------
    # Memory decay
    # ------------------------------------------------------------------

    def apply_decay(self, current_time: float, decay_fn):
        """
        Apply memory decay to all observed cells.

        Decay pulls log-odds toward 0 (P=0.5 = unknown).

        Args:
            current_time: Current simulation timestamp.
            decay_fn: Callable(delta_t, visit_count) -> retention in [0, 1].
                      retention=1.0 means no forgetting.
                      retention=0.0 means fully forgotten.
        """
        # Only decay cells that have been observed
        observed_mask = self.last_observed >= 0

        if not np.any(observed_mask):
            return

        # Compute time since last observation
        delta_t = np.where(observed_mask,
                           current_time - self.last_observed,
                           0.0)

        # Compute retention factor per cell
        # Vectorized: apply decay function
        retention = np.ones_like(self.log_odds)
        observed_indices = np.where(observed_mask)

        for idx in range(len(observed_indices[0])):
            r, c = observed_indices[0][idx], observed_indices[1][idx]
            dt = delta_t[r, c]
            vc = self.visit_count[r, c]
            if dt > 0:
                retention[r, c] = decay_fn(dt, vc)

        # Apply decay: log_odds *= retention (pulls toward 0)
        self.log_odds[observed_mask] *= retention[observed_mask]

    def apply_decay_vectorized(self, current_time: float, decay_fn_vectorized):
        """
        Vectorized version of apply_decay for performance.

        Args:
            current_time: Current simulation timestamp.
            decay_fn_vectorized: Callable(delta_t_array, visit_count_array)
                                -> retention_array. All arrays same shape.
        """
        observed_mask = self.last_observed >= 0
        if not np.any(observed_mask):
            return

        delta_t = np.where(observed_mask,
                           current_time - self.last_observed, 0.0)

        retention = decay_fn_vectorized(delta_t, self.visit_count)
        retention = np.where(observed_mask & (delta_t > 0), retention, 1.0)

        self.log_odds *= retention

    # ------------------------------------------------------------------
    # Probability / uncertainty maps
    # ------------------------------------------------------------------

    def get_probability_map(self) -> np.ndarray:
        """
        Convert log-odds to probability [0, 1] grid.

        P(occupied) = 1 / (1 + exp(-log_odds))
        """
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def get_uncertainty_map(self) -> np.ndarray:
        """
        Return per-cell entropy as uncertainty measure.

        H = -p*log(p) - (1-p)*log(1-p)
        Range: 0 (certain) to ln(2) ≈ 0.693 (maximum uncertainty).
        """
        p = self.get_probability_map()
        p = np.clip(p, 1e-6, 1.0 - 1e-6)  # Avoid log(0)
        entropy = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
        return entropy

    def get_binary_map(self, threshold: float = 0.5) -> np.ndarray:
        """
        Return binary occupancy map (True=occupied, False=free).

        Cells with P(occupied) >= threshold are marked occupied.
        """
        return self.get_probability_map() >= threshold

    # ------------------------------------------------------------------
    # Memory statistics
    # ------------------------------------------------------------------

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Report cells in different certainty states.

        Returns dict with counts of:
            - certain_occupied: |log_odds| > uncertainty_threshold and positive
            - certain_free: |log_odds| > uncertainty_threshold and negative
            - uncertain: |log_odds| <= uncertainty_threshold
            - never_observed: last_observed == -1
        """
        threshold = 0.5  # log-odds threshold for "certain"
        abs_lo = np.abs(self.log_odds)
        never = self.last_observed < 0

        certain_occ = int(np.sum((self.log_odds > threshold) & ~never))
        certain_free = int(np.sum((self.log_odds < -threshold) & ~never))
        uncertain = int(np.sum((abs_lo <= threshold) & ~never))
        never_observed = int(np.sum(never))

        return {
            "certain_occupied": certain_occ,
            "certain_free": certain_free,
            "uncertain": uncertain,
            "never_observed": never_observed,
            "total_observed": certain_occ + certain_free + uncertain,
            "total_cells": self.log_odds.size,
        }

    def get_map_entropy(self) -> float:
        """Return total Shannon entropy of the map."""
        return float(np.sum(self.get_uncertainty_map()))

    def get_coverage(self, grid_world) -> float:
        """
        Return fraction of free cells that have been observed.

        Args:
            grid_world: Ground truth GridWorld for reference.
        """
        total_free = grid_world.get_free_cells_count()
        if total_free == 0:
            return 1.0
        observed = np.sum(self.last_observed >= 0)
        return float(min(observed / total_free, 1.0))

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world (x, y) to grid (row, col)."""
        col = int(x / self.resolution)
        row = self.grid_config.height_cells - 1 - int(y / self.resolution)
        return (row, col)

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid (row, col) to world (x, y) at cell center."""
        x = (col + 0.5) * self.resolution
        y = (self.grid_config.height_cells - 1 - row + 0.5) * self.resolution
        return (x, y)

    def _in_bounds(self, row: int, col: int) -> bool:
        """Check if grid indices are valid."""
        return (0 <= row < self.grid_config.height_cells and
                0 <= col < self.grid_config.width_cells)

    # ------------------------------------------------------------------
    # Scan likelihood (for particle filter weighting)
    # ------------------------------------------------------------------

    def scan_likelihood(self, pose: Tuple[float, float, float],
                        scan: np.ndarray) -> float:
        """
        Compute the likelihood of a scan given this map and a pose.

        Uses a simplified beam endpoint model: check if hit points
        correspond to occupied cells in the grid.

        Args:
            pose: Hypothesized pose (x, y, θ).
            scan: LiDAR scan array (N, 2) of [angle, distance].

        Returns:
            Log-likelihood score (higher = better match).
        """
        x, y, _ = pose
        log_likelihood = 0.0
        hits = 0

        for i in range(len(scan)):
            angle = scan[i, 0]
            distance = scan[i, 1]

            if distance >= self.lidar_config.max_range - 0.05:
                continue  # Skip max-range readings

            # Hit point in world coordinates
            hx = x + distance * np.cos(angle)
            hy = y + distance * np.sin(angle)

            row, col = self.world_to_grid(hx, hy)
            if not self._in_bounds(row, col):
                log_likelihood -= 1.0
                continue

            # Higher log-odds at hit point = better match
            lo = self.log_odds[row, col]
            if lo > 0:
                log_likelihood += min(lo, 2.0)  # Reward: occupied where expected
            else:
                log_likelihood -= 0.5  # Penalty: free where expected occupied

            hits += 1

        # Normalize by number of valid hits
        if hits > 0:
            log_likelihood /= hits

        return log_likelihood

    # ------------------------------------------------------------------
    # Copy / reset
    # ------------------------------------------------------------------

    def copy(self) -> "DecayingOccupancyGrid":
        """Create a deep copy of this grid."""
        new_grid = DecayingOccupancyGrid(self.grid_config, self.lidar_config)
        new_grid.log_odds = self.log_odds.copy()
        new_grid.last_observed = self.last_observed.copy()
        new_grid.visit_count = self.visit_count.copy()
        return new_grid

    def reset(self):
        """Reset grid to unknown state."""
        self.log_odds.fill(self.grid_config.log_odds_prior)
        self.last_observed.fill(-1.0)
        self.visit_count.fill(0)

    # ------------------------------------------------------------------
    # Bresenham
    # ------------------------------------------------------------------

    @staticmethod
    def _bresenham(r0: int, c0: int, r1: int, c1: int):
        """Bresenham's line from (r0,c0) to (r1,c1)."""
        cells = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc

        r, c = r0, c0
        max_steps = dr + dc + 1

        for _ in range(max_steps):
            cells.append((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

        return cells

    @property
    def shape(self) -> Tuple[int, int]:
        return self.log_odds.shape

    def __repr__(self) -> str:
        usage = self.get_memory_usage()
        return (
            f"DecayingOccupancyGrid(shape={self.shape}, "
            f"observed={usage['total_observed']}, "
            f"never={usage['never_observed']})"
        )
