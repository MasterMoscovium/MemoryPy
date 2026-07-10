"""
Simulated 2D LiDAR sensor using Bresenham ray-casting.

Provides range measurements by casting rays through an occupancy grid.
Supports configurable FOV, beam count, max range, and Gaussian noise.
"""

import numpy as np
from typing import Tuple, Optional, List
from config.settings import LidarConfig


class LidarSensor:
    """
    Simulated 2D LiDAR sensor.

    Casts rays through a GridWorld to produce distance measurements,
    simulating a planar laser range finder.

    Each scan returns an array of (angle, distance) pairs.
    """

    def __init__(self, config: LidarConfig,
                 rng: Optional[np.random.Generator] = None):
        """
        Args:
            config: LiDAR configuration parameters.
            rng: Random number generator for noise.
        """
        self.config = config
        self.rng = rng or np.random.default_rng()

        # Precompute beam angles relative to robot heading
        self.beam_angles = np.linspace(
            config.angle_min, config.angle_max, config.num_beams
        )

    def scan(self, pose: Tuple[float, float, float],
             grid_world) -> np.ndarray:
        """
        Perform a LiDAR scan from the given pose.

        Args:
            pose: Robot pose (x, y, θ) in world coordinates.
            grid_world: GridWorld instance to cast rays against.

        Returns:
            Array of shape (num_beams, 2) where each row is
            [angle_global, distance]. Angle is in world frame.
            Distance = max_range if no hit.
        """
        x, y, theta = pose
        results = np.zeros((self.config.num_beams, 2))

        for i, beam_angle in enumerate(self.beam_angles):
            global_angle = theta + beam_angle
            distance = self._cast_ray(x, y, global_angle, grid_world)

            # Add Gaussian noise
            if distance < self.config.max_range:
                distance += self.rng.normal(0, self.config.noise_std)
                distance = np.clip(distance, self.config.min_range,
                                   self.config.max_range)

            results[i, 0] = global_angle
            results[i, 1] = distance

        return results

    def scan_to_points(self, pose: Tuple[float, float, float],
                       scan: np.ndarray) -> np.ndarray:
        """
        Convert scan distances to 2D point cloud in world frame.

        Args:
            pose: Robot pose (x, y, θ).
            scan: Scan array from self.scan().

        Returns:
            Array of shape (N, 2) with (x, y) points for valid hits.
            Points at max_range are excluded.
        """
        x, y, _ = pose
        points = []
        for angle, dist in scan:
            if dist < self.config.max_range - 0.01:
                px = x + dist * np.cos(angle)
                py = y + dist * np.sin(angle)
                points.append([px, py])

        if len(points) == 0:
            return np.empty((0, 2))
        return np.array(points)

    # ------------------------------------------------------------------
    # Ray casting
    # ------------------------------------------------------------------

    def _cast_ray(self, x0: float, y0: float, angle: float,
                  grid_world) -> float:
        """
        Cast a single ray using Bresenham's line algorithm.

        Args:
            x0, y0: Ray origin in world coordinates.
            angle: Ray direction (radians, world frame).
            grid_world: GridWorld to check occupancy against.

        Returns:
            Distance to first hit, or max_range if no hit.
        """
        resolution = grid_world.resolution
        max_range = self.config.max_range

        # Endpoint of ray at max range
        x1 = x0 + max_range * np.cos(angle)
        y1 = y0 + max_range * np.sin(angle)

        # Convert to grid coordinates
        r0, c0 = grid_world.world_to_grid(x0, y0)
        r1, c1 = grid_world.world_to_grid(x1, y1)

        # Bresenham's line algorithm
        cells = self._bresenham(r0, c0, r1, c1)

        for row, col in cells:
            # Skip origin cell
            if row == r0 and col == c0:
                continue

            # Check bounds
            if not grid_world.is_in_bounds_grid(row, col):
                # Hit boundary — compute distance to boundary
                wx, wy = grid_world.grid_to_world(row, col)
                dist = np.sqrt((wx - x0)**2 + (wy - y0)**2)
                return min(dist, max_range)

            # Check occupancy
            if grid_world.grid[row, col] > 0.5:
                # Hit an occupied cell
                wx, wy = grid_world.grid_to_world(row, col)
                dist = np.sqrt((wx - x0)**2 + (wy - y0)**2)
                return max(dist, self.config.min_range)

        return max_range

    @staticmethod
    def _bresenham(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
        """
        Bresenham's line algorithm — returns list of (row, col) cells
        along the line from (r0,c0) to (r1,c1).
        """
        cells = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc

        r, c = r0, c0
        max_steps = dr + dc + 1  # Safety limit

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

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_beam_endpoints(self, pose: Tuple[float, float, float],
                           scan: np.ndarray) -> np.ndarray:
        """
        Get the (x, y) endpoint of each beam (including max-range misses).

        Useful for visualization.

        Returns:
            Array of shape (num_beams, 2).
        """
        x, y, _ = pose
        endpoints = np.zeros((len(scan), 2))
        for i, (angle, dist) in enumerate(scan):
            endpoints[i, 0] = x + dist * np.cos(angle)
            endpoints[i, 1] = y + dist * np.sin(angle)
        return endpoints

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"LidarSensor(range={cfg.max_range}m, "
            f"fov={np.degrees(cfg.fov):.0f}°, "
            f"beams={cfg.num_beams}, "
            f"noise_σ={cfg.noise_std}m)"
        )
