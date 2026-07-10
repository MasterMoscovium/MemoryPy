"""
FastSLAM 1.0 particle filter for simultaneous localization and mapping.

Each particle maintains a pose hypothesis. All particles share a single
occupancy grid (updated using the best particle's pose) for computational
tractability. Particles are weighted by scan likelihood and resampled
when the effective sample size drops below a threshold.
"""

import numpy as np
from typing import Tuple, Optional, List
from config.settings import ParticleFilterConfig, RobotConfig, GridConfig, LidarConfig
from slam.occupancy_grid import DecayingOccupancyGrid


class Particle:
    """A single particle: pose + weight."""
    __slots__ = ['x', 'y', 'theta', 'weight']

    def __init__(self, x: float, y: float, theta: float, weight: float = 1.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

    @property
    def pose(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.theta)


class ParticleFilter:
    """
    Particle filter for robot localization within SLAM.

    Architecture (simplified FastSLAM):
        - N particles each hold a pose hypothesis
        - A shared occupancy grid is updated using the best particle's pose
        - Particles are weighted by scan likelihood against the shared grid
        - Resampling when ESS < threshold

    This is more tractable than full FastSLAM (where each particle has
    its own grid) while still providing useful pose diversity.
    """

    def __init__(self, config: ParticleFilterConfig,
                 robot_config: RobotConfig,
                 grid_config: GridConfig,
                 lidar_config: LidarConfig,
                 initial_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 rng: Optional[np.random.Generator] = None):
        self.config = config
        self.robot_config = robot_config
        self.rng = rng or np.random.default_rng()

        # Initialize particles around the initial pose
        self.particles: List[Particle] = []
        for _ in range(config.num_particles):
            x = initial_pose[0] + self.rng.normal(0, config.initial_spread_xy)
            y = initial_pose[1] + self.rng.normal(0, config.initial_spread_xy)
            theta = initial_pose[2] + self.rng.normal(0, config.initial_spread_theta)
            self.particles.append(Particle(x, y, theta, 1.0 / config.num_particles))

        # Shared occupancy grid
        self.grid = DecayingOccupancyGrid(grid_config, lidar_config)

        # Best estimate tracking
        self._best_particle_idx = 0

    # ------------------------------------------------------------------
    # Main SLAM step
    # ------------------------------------------------------------------

    def update(self, v: float, omega: float, scan: np.ndarray,
               timestamp: float) -> Tuple[float, float, float]:
        """
        Full SLAM update cycle: predict → weight → resample → map update.

        Args:
            v: Commanded linear velocity.
            omega: Commanded angular velocity.
            scan: LiDAR scan array (N, 2) of [angle, distance].
            timestamp: Current simulation time.

        Returns:
            Best pose estimate (x, y, θ).
        """
        # Step 1: Propagate particles through motion model
        self._predict(v, omega)

        # Step 2: Weight particles by scan likelihood
        self._weight(scan)

        # Step 3: Resample if needed
        self._resample_if_needed()

        # Step 4: Update map using best particle's pose
        best_pose = self.get_best_pose()
        self.grid.update(best_pose, scan, timestamp)

        return best_pose

    # ------------------------------------------------------------------
    # Prediction (motion model)
    # ------------------------------------------------------------------

    def _predict(self, v: float, omega: float):
        """Propagate each particle through the noisy motion model."""
        dt = self.robot_config.dt
        cfg = self.robot_config

        for p in self.particles:
            # Add noise to velocities
            v_var = cfg.alpha1 * abs(v) + cfg.alpha2 * abs(omega)
            omega_var = cfg.alpha3 * abs(omega) + cfg.alpha4 * abs(v)

            v_hat = v + self.rng.normal(0, max(v_var, 1e-6))
            omega_hat = omega + self.rng.normal(0, max(omega_var, 1e-6))

            # Apply motion model
            if abs(omega_hat) < 1e-6:
                p.x += v_hat * np.cos(p.theta) * dt
                p.y += v_hat * np.sin(p.theta) * dt
            else:
                r = v_hat / omega_hat
                p.x += r * (-np.sin(p.theta) + np.sin(p.theta + omega_hat * dt))
                p.y += r * (np.cos(p.theta) - np.cos(p.theta + omega_hat * dt))
                p.theta += omega_hat * dt

            # Normalize angle
            while p.theta > np.pi:
                p.theta -= 2.0 * np.pi
            while p.theta < -np.pi:
                p.theta += 2.0 * np.pi

    # ------------------------------------------------------------------
    # Weighting (scan likelihood)
    # ------------------------------------------------------------------

    def _weight(self, scan: np.ndarray):
        """Weight each particle by how well the scan matches the grid."""
        max_log_ll = -float('inf')
        log_likelihoods = []

        for i, p in enumerate(self.particles):
            ll = self.grid.scan_likelihood(p.pose, scan)
            log_likelihoods.append(ll)
            if ll > max_log_ll:
                max_log_ll = ll
                self._best_particle_idx = i

        # Convert to weights using log-sum-exp for numerical stability
        log_likelihoods = np.array(log_likelihoods)
        # Shift for numerical stability
        shifted = log_likelihoods - max_log_ll
        weights = np.exp(shifted)

        # Normalize
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights /= weight_sum
        else:
            weights = np.ones(len(self.particles)) / len(self.particles)

        for i, p in enumerate(self.particles):
            p.weight = weights[i]

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def _resample_if_needed(self):
        """Resample particles if effective sample size is too low."""
        ess = self.effective_sample_size()
        threshold = self.config.resample_threshold * len(self.particles)

        if ess < threshold:
            self._low_variance_resample()

    def _low_variance_resample(self):
        """Low-variance resampling (systematic resampling)."""
        N = len(self.particles)
        weights = np.array([p.weight for p in self.particles])

        # Ensure weights sum to 1
        w_sum = np.sum(weights)
        if w_sum > 0:
            weights /= w_sum
        else:
            weights = np.ones(N) / N

        # Cumulative sum
        cumsum = np.cumsum(weights)

        # Random start point
        r = self.rng.uniform(0, 1.0 / N)
        new_particles = []

        idx = 0
        for i in range(N):
            u = r + i / N
            while idx < N - 1 and cumsum[idx] < u:
                idx += 1

            old = self.particles[idx]
            new_particles.append(Particle(
                old.x + self.rng.normal(0, 0.01),  # Small jitter to avoid collapse
                old.y + self.rng.normal(0, 0.01),
                old.theta + self.rng.normal(0, 0.005),
                1.0 / N
            ))

        self.particles = new_particles
        self._best_particle_idx = 0

    # ------------------------------------------------------------------
    # Pose estimation
    # ------------------------------------------------------------------

    def get_best_pose(self) -> Tuple[float, float, float]:
        """Return the pose of the highest-weight particle."""
        if 0 <= self._best_particle_idx < len(self.particles):
            return self.particles[self._best_particle_idx].pose
        return self.get_weighted_mean_pose()

    def get_weighted_mean_pose(self) -> Tuple[float, float, float]:
        """Return the weighted mean pose across all particles."""
        weights = np.array([p.weight for p in self.particles])
        w_sum = np.sum(weights)
        if w_sum == 0:
            weights = np.ones(len(self.particles)) / len(self.particles)
            w_sum = 1.0

        x = sum(p.x * p.weight for p in self.particles) / w_sum
        y = sum(p.y * p.weight for p in self.particles) / w_sum

        # Circular mean for angle
        sin_sum = sum(p.weight * np.sin(p.theta) for p in self.particles) / w_sum
        cos_sum = sum(p.weight * np.cos(p.theta) for p in self.particles) / w_sum
        theta = np.arctan2(sin_sum, cos_sum)

        return (x, y, theta)

    def get_particle_poses(self) -> np.ndarray:
        """Return all particle poses as (N, 3) array."""
        return np.array([[p.x, p.y, p.theta] for p in self.particles])

    def get_particle_weights(self) -> np.ndarray:
        """Return all particle weights as (N,) array."""
        return np.array([p.weight for p in self.particles])

    def effective_sample_size(self) -> float:
        """Compute effective sample size: ESS = 1 / Σ(w²)."""
        weights = np.array([p.weight for p in self.particles])
        w_sum = np.sum(weights)
        if w_sum == 0:
            return len(self.particles)
        weights = weights / w_sum
        return 1.0 / np.sum(weights ** 2)

    # ------------------------------------------------------------------
    # Map access
    # ------------------------------------------------------------------

    def get_occupancy_grid(self) -> DecayingOccupancyGrid:
        """Return the shared occupancy grid."""
        return self.grid

    def get_probability_map(self) -> np.ndarray:
        """Return the probability map from the shared grid."""
        return self.grid.get_probability_map()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, initial_pose: Tuple[float, float, float]):
        """Reset all particles around a new initial pose."""
        cfg = self.config
        self.particles = []
        for _ in range(cfg.num_particles):
            x = initial_pose[0] + self.rng.normal(0, cfg.initial_spread_xy)
            y = initial_pose[1] + self.rng.normal(0, cfg.initial_spread_xy)
            theta = initial_pose[2] + self.rng.normal(0, cfg.initial_spread_theta)
            self.particles.append(Particle(x, y, theta, 1.0 / cfg.num_particles))
        self.grid.reset()
        self._best_particle_idx = 0

    def __repr__(self) -> str:
        ess = self.effective_sample_size()
        best = self.get_best_pose()
        return (
            f"ParticleFilter(n={len(self.particles)}, "
            f"ESS={ess:.1f}, "
            f"best=({best[0]:.2f}, {best[1]:.2f}, {np.degrees(best[2]):.1f}°))"
        )
