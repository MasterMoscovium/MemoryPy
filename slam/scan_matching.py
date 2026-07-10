"""
Simple ICP (Iterative Closest Point) scan matching for pose refinement.

Aligns a new point cloud (from current scan) against a reference
point cloud (from the map or previous scan) to estimate the
rigid-body transformation that best aligns them.
"""

import numpy as np
from typing import Tuple, Optional


class ScanMatcher:
    """
    ICP-based scan matcher for 2D point clouds.

    Used to refine robot pose estimates by aligning the current
    LiDAR scan against the expected scan from the occupancy grid.
    """

    def __init__(self, max_iterations: int = 20,
                 convergence_threshold: float = 1e-4,
                 max_correspondence_dist: float = 1.0):
        """
        Args:
            max_iterations: Maximum ICP iterations.
            convergence_threshold: Stop if transform change < this.
            max_correspondence_dist: Reject point pairs farther than this (m).
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.max_correspondence_dist = max_correspondence_dist

    def match(self, source: np.ndarray, target: np.ndarray
              ) -> Tuple[np.ndarray, float, bool]:
        """
        Align source points to target points using ICP.

        Args:
            source: (N, 2) array of points to be transformed.
            target: (M, 2) array of reference points.

        Returns:
            Tuple of:
                - transform: (3,) array [dx, dy, dtheta] correction
                - error: Mean squared error of final alignment
                - converged: Whether ICP converged within max_iterations
        """
        if len(source) < 3 or len(target) < 3:
            return np.zeros(3), float('inf'), False

        # Work with copies
        src = source.copy()
        cumulative_R = np.eye(2)
        cumulative_t = np.zeros(2)
        prev_error = float('inf')
        converged = False

        for iteration in range(self.max_iterations):
            # Step 1: Find closest points in target for each source point
            correspondences, distances = self._find_correspondences(src, target)

            # Filter by max correspondence distance
            valid = distances < self.max_correspondence_dist
            if np.sum(valid) < 3:
                break

            src_valid = src[valid]
            tgt_valid = correspondences[valid]

            # Step 2: Compute optimal rotation + translation
            R, t = self._compute_transform(src_valid, tgt_valid)

            # Step 3: Apply transform to source
            src = (R @ src.T).T + t
            cumulative_R = R @ cumulative_R
            cumulative_t = R @ cumulative_t + t

            # Check convergence
            mean_error = float(np.mean(distances[valid] ** 2))
            if abs(prev_error - mean_error) < self.convergence_threshold:
                converged = True
                break
            prev_error = mean_error

        # Extract (dx, dy, dtheta) from cumulative transform
        dtheta = np.arctan2(cumulative_R[1, 0], cumulative_R[0, 0])
        dx = cumulative_t[0]
        dy = cumulative_t[1]

        return np.array([dx, dy, dtheta]), prev_error, converged

    def refine_pose(self, pose: Tuple[float, float, float],
                    scan_points: np.ndarray,
                    map_points: np.ndarray
                    ) -> Tuple[Tuple[float, float, float], float, bool]:
        """
        Refine a robot pose by matching scan points to map points.

        Args:
            pose: Current pose estimate (x, y, θ).
            scan_points: (N, 2) points from current scan (world frame).
            map_points: (M, 2) points extracted from the map.

        Returns:
            Tuple of:
                - refined_pose: (x, y, θ) after ICP correction
                - error: Alignment error
                - converged: Whether ICP converged
        """
        if len(scan_points) < 3 or len(map_points) < 3:
            return pose, float('inf'), False

        correction, error, converged = self.match(scan_points, map_points)

        x, y, theta = pose
        refined_x = x + correction[0]
        refined_y = y + correction[1]
        refined_theta = theta + correction[2]

        # Normalize angle
        while refined_theta > np.pi:
            refined_theta -= 2 * np.pi
        while refined_theta < -np.pi:
            refined_theta += 2 * np.pi

        return (refined_x, refined_y, refined_theta), error, converged

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _find_correspondences(source: np.ndarray, target: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest neighbor in target for each source point.

        Returns:
            correspondences: (N, 2) matched target points
            distances: (N,) distances to matches
        """
        # Brute-force nearest neighbor (sufficient for our point cloud sizes)
        correspondences = np.zeros_like(source)
        distances = np.zeros(len(source))

        for i, sp in enumerate(source):
            diffs = target - sp
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            min_idx = np.argmin(dists)
            correspondences[i] = target[min_idx]
            distances[i] = dists[min_idx]

        return correspondences, distances

    @staticmethod
    def _compute_transform(source: np.ndarray, target: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optimal rigid transform (R, t) to align source to target.

        Uses SVD-based approach:
            target = R @ source + t

        Returns:
            R: (2, 2) rotation matrix
            t: (2,) translation vector
        """
        # Centroids
        src_centroid = np.mean(source, axis=0)
        tgt_centroid = np.mean(target, axis=0)

        # Center the points
        src_centered = source - src_centroid
        tgt_centered = target - tgt_centroid

        # Cross-covariance matrix
        H = src_centered.T @ tgt_centered

        # SVD
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Translation
        t = tgt_centroid - R @ src_centroid

        return R, t

    def extract_map_points(self, occupancy_grid, pose: Tuple[float, float, float],
                           radius: float = 8.0,
                           threshold: float = 0.6) -> np.ndarray:
        """
        Extract occupied points from the occupancy grid near a given pose.

        Args:
            occupancy_grid: DecayingOccupancyGrid instance.
            pose: Robot pose (x, y, θ).
            radius: Search radius in meters.
            threshold: Probability threshold for "occupied".

        Returns:
            (M, 2) array of occupied cell centers in world coordinates.
        """
        x, y, _ = pose
        prob_map = occupancy_grid.get_probability_map()
        points = []

        # Convert search radius to grid cells
        r_cells = int(radius / occupancy_grid.resolution)
        center_r, center_c = occupancy_grid.world_to_grid(x, y)

        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                r = center_r + dr
                c = center_c + dc
                if occupancy_grid._in_bounds(r, c):
                    if prob_map[r, c] >= threshold:
                        wx, wy = occupancy_grid.grid_to_world(r, c)
                        dist = np.sqrt((wx - x)**2 + (wy - y)**2)
                        if dist <= radius:
                            points.append([wx, wy])

        if len(points) == 0:
            return np.empty((0, 2))
        return np.array(points)

    def __repr__(self) -> str:
        return (
            f"ScanMatcher(max_iter={self.max_iterations}, "
            f"conv_thresh={self.convergence_threshold}, "
            f"max_corr_dist={self.max_correspondence_dist}m)"
        )
