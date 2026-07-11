"""
Central configuration for the MemoryPy simulation.

All tunable parameters are defined as dataclasses here.
Modify these to change simulation behavior without touching code.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum
import math


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DecayModelType(Enum):
    """Available memory decay models."""
    NONE = "none"                    # No decay (control baseline)
    EXPONENTIAL = "exponential"      # R = e^(-λ·Δt)
    POWER_LAW = "power_law"          # R = (Δt+1)^(-β)
    ADAPTIVE = "adaptive"            # R = e^(-λ/S·Δt), S grows with visits
    AGGRESSIVE = "aggressive"        # Exponential with very high λ
    THRESHOLD = "threshold"          # Binary: forget if certainty < τ


class NavigationStrategy(Enum):
    """Goal-selection weight presets."""
    EXPLORATION_HEAVY = "exploration_heavy"
    BALANCED = "balanced"
    DECAY_PRIORITY = "decay_priority"
    GREEDY = "greedy"
    RANDOM = "random"


# ---------------------------------------------------------------------------
# Grid Configuration
# ---------------------------------------------------------------------------

@dataclass
class GridConfig:
    """Occupancy grid parameters."""
    resolution: float = 0.1           # Meters per cell
    width_m: float = 20.0             # Environment width in meters
    height_m: float = 20.0            # Environment height in meters
    log_odds_free: float = -2.0       # Log-odds value for "definitely free"
    log_odds_occupied: float = 2.0    # Log-odds value for "definitely occupied"
    log_odds_prior: float = 0.0       # Prior (unknown) log-odds
    log_odds_max: float = 5.0         # Clamp upper bound
    log_odds_min: float = -5.0        # Clamp lower bound

    @property
    def width_cells(self) -> int:
        return int(self.width_m / self.resolution)

    @property
    def height_cells(self) -> int:
        return int(self.height_m / self.resolution)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height_cells, self.width_cells)


# ---------------------------------------------------------------------------
# Robot Configuration
# ---------------------------------------------------------------------------

@dataclass
class RobotConfig:
    """Differential-drive robot parameters."""
    max_linear_velocity: float = 1.5       # m/s
    max_angular_velocity: float = 2.5      # rad/s
    dt: float = 0.1                         # Simulation timestep (seconds)

    # Odometry noise (standard deviations)
    alpha1: float = 0.05    # Rotation noise from rotation
    alpha2: float = 0.01    # Rotation noise from translation
    alpha3: float = 0.05    # Translation noise from translation
    alpha4: float = 0.01    # Translation noise from rotation

    # Physical dimensions
    radius: float = 0.15     # Robot radius in meters (for collision)

    # Initial pose
    start_x: float = 0.0
    start_y: float = 0.0
    start_theta: float = 0.0


# ---------------------------------------------------------------------------
# LiDAR Configuration
# ---------------------------------------------------------------------------

@dataclass
class LidarConfig:
    """Simulated 2D LiDAR parameters."""
    max_range: float = 8.0           # Maximum sensing range (meters)
    min_range: float = 0.1           # Minimum sensing range (meters)
    fov: float = 4.712389            # Field of view in radians (270°)
    num_beams: int = 180             # Number of laser beams
    noise_std: float = 0.02          # Gaussian noise σ on distance (meters)

    # Inverse sensor model parameters
    p_occupied: float = 0.7          # P(occupied | hit)
    p_free: float = 0.3              # P(free | pass-through)
    wall_thickness: int = 2          # Cells around hit to mark occupied

    @property
    def angle_increment(self) -> float:
        """Angle between consecutive beams in radians."""
        return self.fov / max(self.num_beams - 1, 1)

    @property
    def angle_min(self) -> float:
        """Start angle relative to robot heading."""
        return -self.fov / 2.0

    @property
    def angle_max(self) -> float:
        """End angle relative to robot heading."""
        return self.fov / 2.0


# ---------------------------------------------------------------------------
# Decay Configuration
# ---------------------------------------------------------------------------

@dataclass
class DecayConfig:
    """Memory decay parameters."""
    model_type: DecayModelType = DecayModelType.EXPONENTIAL

    # Exponential decay: R = e^(-lambda * dt)
    decay_lambda: float = 0.01

    # Power-law decay: R = (dt + 1)^(-beta)
    power_beta: float = 0.5

    # Adaptive decay: S = 1 + alpha * visits, R = e^(-lambda/S * dt)
    adaptive_alpha: float = 0.3      # How much each visit increases stability
    stability_max: float = 10.0      # Maximum stability value

    # Aggressive decay (exponential with high lambda)
    aggressive_lambda: float = 0.2

    # Threshold decay
    certainty_threshold: float = 0.3  # |log_odds| below this → reset to unknown

    # General decay settings
    decay_interval: int = 10          # Apply decay every N timesteps
    min_retention: float = 0.01       # Minimum retention before cell is "forgotten"
    uncertainty_threshold: float = 0.2  # |log_odds| below this = "uncertain"


# ---------------------------------------------------------------------------
# Navigation Configuration
# ---------------------------------------------------------------------------

@dataclass
class NavigationConfig:
    """Path planning and exploration parameters."""
    strategy: NavigationStrategy = NavigationStrategy.BALANCED

    # Utility function weights (for BALANCED strategy)
    w_info_gain: float = 1.0         # Weight for information gain
    w_decay_urgency: float = 0.5     # Weight for decay urgency
    w_distance: float = 0.3          # Weight for travel distance

    # A* path planner
    obstacle_cost: float = 100.0     # Cost multiplier for occupied cells
    uncertainty_cost: float = 5.0    # Cost multiplier for uncertain cells
    base_cost: float = 1.0           # Base traversal cost per cell
    replan_interval: int = 20        # Replan path every N steps

    # Frontier detection
    min_frontier_size: int = 3       # Minimum cells for a valid frontier
    decay_frontier_enabled: bool = True  # Also detect decay frontiers

    # Strategy-specific weight overrides
    STRATEGY_WEIGHTS = {
        NavigationStrategy.EXPLORATION_HEAVY: (2.0, 0.1, 0.3),
        NavigationStrategy.BALANCED:          (1.0, 0.5, 0.3),
        NavigationStrategy.DECAY_PRIORITY:    (0.3, 2.0, 0.3),
        NavigationStrategy.GREEDY:            (1.0, 0.0, 2.0),
        NavigationStrategy.RANDOM:            (0.0, 0.0, 0.0),
    }

    def get_strategy_weights(self) -> Tuple[float, float, float]:
        """Return (w_info, w_decay, w_dist) for the current strategy."""
        if self.strategy == NavigationStrategy.RANDOM:
            return (0.0, 0.0, 0.0)
        return self.STRATEGY_WEIGHTS.get(
            self.strategy,
            (self.w_info_gain, self.w_decay_urgency, self.w_distance)
        )


# ---------------------------------------------------------------------------
# Particle Filter Configuration
# ---------------------------------------------------------------------------

@dataclass
class ParticleFilterConfig:
    """FastSLAM particle filter parameters."""
    num_particles: int = 20          # Number of particles
    resample_threshold: float = 0.5  # ESS ratio threshold for resampling
    initial_spread_xy: float = 0.1   # Initial particle spread (meters)
    initial_spread_theta: float = 0.05  # Initial angular spread (radians)


# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Experiment execution parameters."""
    max_timesteps: int = 2000        # Maximum simulation steps per run
    num_seeds: int = 5               # Repetitions with different seeds
    seeds: Optional[List[int]] = None  # Specific seeds (auto-generated if None)
    metrics_interval: int = 50       # Record metrics every N steps
    save_maps: bool = False          # Save occupancy grid snapshots
    map_save_interval: int = 500     # Save map every N steps
    results_dir: str = "results"     # Output directory

    def get_seeds(self) -> List[int]:
        """Return list of seeds for this experiment."""
        if self.seeds is not None:
            return self.seeds
        return list(range(42, 42 + self.num_seeds))


# ---------------------------------------------------------------------------
# Top-Level Simulation Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Master configuration combining all sub-configs."""
    grid: GridConfig = field(default_factory=GridConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    particle_filter: ParticleFilterConfig = field(default_factory=ParticleFilterConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Environment
    environment_name: str = "simple_room"
    enable_dynamic_obstacles: bool = False

    # Visualization
    visualize: bool = False
    visualization_fps: int = 30

    @classmethod
    def for_environment(cls, env_name: str, **kwargs) -> "SimulationConfig":
        """Create a config pre-tuned for a specific environment."""
        env_sizes = {
            "simple_room":       (20.0, 20.0),
            "office_maze":       (40.0, 30.0),
            "open_field":        (50.0, 50.0),
            "dynamic_obstacles": (30.0, 30.0),
        }
        w, h = env_sizes.get(env_name, (20.0, 20.0))
        grid = GridConfig(width_m=w, height_m=h)

        # Center the robot
        robot = RobotConfig(start_x=w / 2.0, start_y=h / 2.0)

        config = cls(
            grid=grid,
            robot=robot,
            environment_name=env_name,
            **kwargs,
        )
        return config
