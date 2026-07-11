"""
Differential-drive robot kinematics model.

Implements a velocity motion model with configurable Gaussian noise
for realistic odometry simulation. Tracks both noisy (estimated) and
ground-truth poses for evaluation.
"""

import numpy as np
from typing import Tuple, Optional
from config.settings import RobotConfig


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π]."""
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


class DifferentialDriveRobot:
    """
    Simulated differential-drive robot with noisy odometry.

    State: (x, y, θ) in world coordinates.
    Motion model: velocity-based with Gaussian noise parameterized
    by alpha1..alpha4 (see Probabilistic Robotics, Thrun et al.).
    """

    def __init__(self, config: RobotConfig, rng: Optional[np.random.Generator] = None):
        """
        Initialize robot at configured start pose.

        Args:
            config: Robot configuration parameters.
            rng: Random number generator for reproducibility.
        """
        self.config = config
        self.rng = rng or np.random.default_rng()

        # Ground-truth pose (used for sensor simulation & evaluation)
        self.true_x = config.start_x
        self.true_y = config.start_y
        self.true_theta = config.start_theta

        # Estimated pose (what the robot "thinks" — affected by noise)
        self.est_x = config.start_x
        self.est_y = config.start_y
        self.est_theta = config.start_theta

        # Trajectory history
        self.true_trajectory = [(self.true_x, self.true_y, self.true_theta)]
        self.est_trajectory = [(self.est_x, self.est_y, self.est_theta)]

        # Cumulative distance traveled
        self.total_distance = 0.0

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move(self, v: float, omega: float,
             collision_fn=None) -> Tuple[float, float]:
        """
        Move the robot with given linear and angular velocity.

        The ground-truth pose is updated deterministically.
        The estimated pose is updated with added noise.

        Args:
            v: Linear velocity (m/s). Clamped to max.
            omega: Angular velocity (rad/s). Clamped to max.
            collision_fn: Optional callable(x, y, radius) -> bool.
                         If provided, movement is rejected on collision.

        Returns:
            (v_noisy, omega_noisy): The noisy velocities actually "measured"
            by the robot's odometry.
        """
        dt = self.config.dt
        cfg = self.config

        # Clamp velocities
        v = np.clip(v, -cfg.max_linear_velocity, cfg.max_linear_velocity)
        omega = np.clip(omega, -cfg.max_angular_velocity, cfg.max_angular_velocity)

        # --- Ground-truth update (deterministic) ---
        new_true_x, new_true_y, new_true_theta = self._apply_motion(
            self.true_x, self.true_y, self.true_theta, v, omega, dt
        )

        # Check collision for ground-truth
        if collision_fn is not None and collision_fn(new_true_x, new_true_y, cfg.radius):
            # Collision: don't move ground truth or estimate, return zero velocity
            v_noisy, omega_noisy = 0.0, 0.0
        else:
            dist = np.sqrt((new_true_x - self.true_x)**2 +
                           (new_true_y - self.true_y)**2)
            self.total_distance += dist
            self.true_x = new_true_x
            self.true_y = new_true_y
            self.true_theta = normalize_angle(new_true_theta)

            # --- Noisy odometry (what the robot "measures") ---
            v_noisy, omega_noisy = self._add_motion_noise(v, omega)

            new_est_x, new_est_y, new_est_theta = self._apply_motion(
                self.est_x, self.est_y, self.est_theta, v_noisy, omega_noisy, dt
            )

            self.est_x = new_est_x
            self.est_y = new_est_y
            self.est_theta = normalize_angle(new_est_theta)

        # Record trajectory
        self.true_trajectory.append((self.true_x, self.true_y, self.true_theta))
        self.est_trajectory.append((self.est_x, self.est_y, self.est_theta))

        return v_noisy, omega_noisy

    def _apply_motion(self, x: float, y: float, theta: float,
                      v: float, omega: float, dt: float
                      ) -> Tuple[float, float, float]:
        """Apply velocity motion model to a pose."""
        if abs(omega) < 1e-6:
            # Straight-line motion
            new_x = x + v * np.cos(theta) * dt
            new_y = y + v * np.sin(theta) * dt
            new_theta = theta
        else:
            # Arc motion
            r = v / omega
            new_x = x + r * (-np.sin(theta) + np.sin(theta + omega * dt))
            new_y = y + r * (np.cos(theta) - np.cos(theta + omega * dt))
            new_theta = theta + omega * dt

        return new_x, new_y, new_theta

    def _add_motion_noise(self, v: float, omega: float
                          ) -> Tuple[float, float]:
        """
        Add noise to velocity commands using the odometry noise model.

        Noise model (Probabilistic Robotics, Table 5.6):
            v_hat = v + sample(α1*|v| + α2*|ω|)
            ω_hat = ω + sample(α3*|ω| + α4*|v|)
        """
        cfg = self.config
        v_var = cfg.alpha1 * abs(v) + cfg.alpha2 * abs(omega)
        omega_var = cfg.alpha3 * abs(omega) + cfg.alpha4 * abs(v)

        v_noisy = v + self.rng.normal(0, max(v_var, 1e-6))
        omega_noisy = omega + self.rng.normal(0, max(omega_var, 1e-6))

        return v_noisy, omega_noisy

    # ------------------------------------------------------------------
    # Pose access
    # ------------------------------------------------------------------

    @property
    def true_pose(self) -> Tuple[float, float, float]:
        """Ground-truth pose (x, y, θ)."""
        return (self.true_x, self.true_y, self.true_theta)

    @property
    def estimated_pose(self) -> Tuple[float, float, float]:
        """Estimated (noisy odometry) pose (x, y, θ)."""
        return (self.est_x, self.est_y, self.est_theta)

    def set_estimated_pose(self, x: float, y: float, theta: float):
        """Override estimated pose (e.g., from SLAM correction)."""
        self.est_x = x
        self.est_y = y
        self.est_theta = normalize_angle(theta)

    def localization_error(self) -> float:
        """Euclidean distance between true and estimated pose."""
        return np.sqrt((self.true_x - self.est_x)**2 +
                       (self.true_y - self.est_y)**2)

    def heading_error(self) -> float:
        """Absolute angular error between true and estimated heading."""
        return abs(normalize_angle(self.true_theta - self.est_theta))

    # ------------------------------------------------------------------
    # Sampling (for particle filter)
    # ------------------------------------------------------------------

    def sample_motion(self, pose: Tuple[float, float, float],
                      v: float, omega: float,
                      rng: Optional[np.random.Generator] = None
                      ) -> Tuple[float, float, float]:
        """
        Sample a new pose from the motion model given a prior pose.

        Used by the particle filter to propagate particles.

        Args:
            pose: Prior pose (x, y, θ).
            v: Commanded linear velocity.
            omega: Commanded angular velocity.
            rng: RNG for this sample (uses self.rng if None).

        Returns:
            Sampled new pose (x, y, θ).
        """
        rng = rng or self.rng
        dt = self.config.dt
        cfg = self.config

        # Add noise to velocities
        v_var = cfg.alpha1 * abs(v) + cfg.alpha2 * abs(omega)
        omega_var = cfg.alpha3 * abs(omega) + cfg.alpha4 * abs(v)

        v_hat = v + rng.normal(0, max(v_var, 1e-6))
        omega_hat = omega + rng.normal(0, max(omega_var, 1e-6))

        x, y, theta = pose
        nx, ny, nt = self._apply_motion(x, y, theta, v_hat, omega_hat, dt)
        return (nx, ny, normalize_angle(nt))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Reset robot to initial pose."""
        cfg = self.config
        self.true_x = cfg.start_x
        self.true_y = cfg.start_y
        self.true_theta = cfg.start_theta
        self.est_x = cfg.start_x
        self.est_y = cfg.start_y
        self.est_theta = cfg.start_theta
        self.true_trajectory = [(self.true_x, self.true_y, self.true_theta)]
        self.est_trajectory = [(self.est_x, self.est_y, self.est_theta)]
        self.total_distance = 0.0

    def __repr__(self) -> str:
        return (
            f"DifferentialDriveRobot("
            f"true=({self.true_x:.2f}, {self.true_y:.2f}, {self.true_theta:.2f}), "
            f"est=({self.est_x:.2f}, {self.est_y:.2f}, {self.est_theta:.2f}), "
            f"dist={self.total_distance:.2f}m)"
        )
