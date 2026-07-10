"""
Experiment configurations — predefined setups for the 5 experiments.

Each experiment is a list of SimulationConfig variants to run,
with specific parameters varied while others are held constant.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from config.settings import (
    SimulationConfig, GridConfig, RobotConfig, LidarConfig,
    DecayConfig, ExperimentConfig, NavigationConfig,
    DecayModelType, NavigationStrategy, ParticleFilterConfig
)


@dataclass
class ExperimentRun:
    """A single experiment run configuration."""
    name: str                      # Human-readable name
    config: SimulationConfig       # Full simulation config
    seed: int                      # Random seed
    tags: Dict[str, Any] = None    # Metadata tags for grouping results

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


def get_experiment_1_runs() -> List[ExperimentRun]:
    """
    Experiment 1: Decay Model Comparison (Core Experiment)

    Compare all 6 decay models across all 4 environments.
    Variables: decay_model (6) × environment (4) × seeds (5) = 120 runs
    """
    runs = []
    environments = ["simple_room", "office_maze", "open_field", "dynamic_obstacles"]
    decay_models = [
        (DecayModelType.NONE, {}),
        (DecayModelType.EXPONENTIAL, {"decay_lambda": 0.01}),
        (DecayModelType.POWER_LAW, {"power_beta": 0.5}),
        (DecayModelType.ADAPTIVE, {"decay_lambda": 0.01, "adaptive_alpha": 0.3}),
        (DecayModelType.AGGRESSIVE, {"aggressive_lambda": 0.2}),
        (DecayModelType.THRESHOLD, {"certainty_threshold": 0.3}),
    ]
    seeds = [42, 43, 44, 45, 46]

    for env_name in environments:
        for decay_type, decay_params in decay_models:
            for seed in seeds:
                cfg = SimulationConfig.for_environment(env_name)
                cfg.decay = DecayConfig(model_type=decay_type, **decay_params)
                cfg.experiment = ExperimentConfig(
                    max_timesteps=2000, metrics_interval=50
                )
                cfg.enable_dynamic_obstacles = (env_name == "dynamic_obstacles")

                runs.append(ExperimentRun(
                    name=f"exp1_{env_name}_{decay_type.value}_s{seed}",
                    config=cfg,
                    seed=seed,
                    tags={
                        "experiment": 1,
                        "environment": env_name,
                        "decay_model": decay_type.value,
                        "seed": seed,
                    }
                ))

    return runs


def get_experiment_2_runs() -> List[ExperimentRun]:
    """
    Experiment 2: Decay Rate Sensitivity Analysis

    Sweep λ for exponential decay in office_maze.
    Variables: λ (8) × seeds (5) = 40 runs
    """
    runs = []
    lambdas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    seeds = [42, 43, 44, 45, 46]

    for lam in lambdas:
        for seed in seeds:
            cfg = SimulationConfig.for_environment("office_maze")
            cfg.decay = DecayConfig(
                model_type=DecayModelType.EXPONENTIAL,
                decay_lambda=lam,
            )
            cfg.experiment = ExperimentConfig(
                max_timesteps=2000, metrics_interval=50
            )

            runs.append(ExperimentRun(
                name=f"exp2_lambda{lam}_s{seed}",
                config=cfg,
                seed=seed,
                tags={
                    "experiment": 2,
                    "decay_lambda": lam,
                    "seed": seed,
                }
            ))

    return runs


def get_experiment_3_runs() -> List[ExperimentRun]:
    """
    Experiment 3: Navigation Strategy Ablation

    Test 5 goal-selection weight presets with adaptive decay in office_maze.
    Variables: strategy (5) × seeds (5) = 25 runs
    """
    runs = []
    strategies = [
        NavigationStrategy.EXPLORATION_HEAVY,
        NavigationStrategy.BALANCED,
        NavigationStrategy.DECAY_PRIORITY,
        NavigationStrategy.GREEDY,
        NavigationStrategy.RANDOM,
    ]
    seeds = [42, 43, 44, 45, 46]

    for strategy in strategies:
        for seed in seeds:
            cfg = SimulationConfig.for_environment("office_maze")
            cfg.decay = DecayConfig(
                model_type=DecayModelType.ADAPTIVE,
                decay_lambda=0.01, adaptive_alpha=0.3,
            )
            cfg.navigation = NavigationConfig(strategy=strategy)
            cfg.experiment = ExperimentConfig(
                max_timesteps=2000, metrics_interval=50
            )

            runs.append(ExperimentRun(
                name=f"exp3_{strategy.value}_s{seed}",
                config=cfg,
                seed=seed,
                tags={
                    "experiment": 3,
                    "strategy": strategy.value,
                    "seed": seed,
                }
            ))

    return runs


def get_experiment_4_runs() -> List[ExperimentRun]:
    """
    Experiment 4: Dynamic Environment Adaptation

    Test decay models in environments with different obstacle change rates.
    Variables: decay (3) × change_freq (3) × seeds (5) = 45 runs
    """
    runs = []
    decay_models = [
        (DecayModelType.NONE, {}),
        (DecayModelType.EXPONENTIAL, {"decay_lambda": 0.01}),
        (DecayModelType.ADAPTIVE, {"decay_lambda": 0.01, "adaptive_alpha": 0.3}),
    ]
    # We use the dynamic_obstacles environment which has built-in events
    # For different frequencies, we adjust the decay interval
    change_configs = [
        ("static", "simple_room", False),
        ("moderate", "dynamic_obstacles", True),
        ("high_change", "dynamic_obstacles", True),
    ]
    seeds = [42, 43, 44, 45, 46]

    for decay_type, decay_params in decay_models:
        for change_name, env_name, dynamic in change_configs:
            for seed in seeds:
                cfg = SimulationConfig.for_environment(env_name)
                cfg.decay = DecayConfig(model_type=decay_type, **decay_params)
                cfg.enable_dynamic_obstacles = dynamic
                cfg.experiment = ExperimentConfig(
                    max_timesteps=2000, metrics_interval=50
                )

                # For high_change, reduce decay interval
                if change_name == "high_change":
                    cfg.decay.decay_interval = 5

                runs.append(ExperimentRun(
                    name=f"exp4_{decay_type.value}_{change_name}_s{seed}",
                    config=cfg,
                    seed=seed,
                    tags={
                        "experiment": 4,
                        "decay_model": decay_type.value,
                        "change_frequency": change_name,
                        "seed": seed,
                    }
                ))

    return runs


def get_experiment_5_runs() -> List[ExperimentRun]:
    """
    Experiment 5: Scalability Analysis

    Measure computational cost across grid sizes.
    Variables: grid_size (4) × decay (2) × seeds (3) = 24 runs
    """
    runs = []
    grid_sizes = [
        (10.0, 10.0, 0.1),    # 100×100
        (20.0, 20.0, 0.1),    # 200×200
        (40.0, 40.0, 0.1),    # 400×400
        (80.0, 80.0, 0.1),    # 800×800
    ]
    decay_models = [
        (DecayModelType.NONE, {}),
        (DecayModelType.EXPONENTIAL, {"decay_lambda": 0.01}),
    ]
    seeds = [42, 43, 44]

    for w, h, res in grid_sizes:
        for decay_type, decay_params in decay_models:
            for seed in seeds:
                grid_cfg = GridConfig(width_m=w, height_m=h, resolution=res)
                robot_cfg = RobotConfig(start_x=w/2, start_y=h/2)

                cfg = SimulationConfig(
                    grid=grid_cfg,
                    robot=robot_cfg,
                    decay=DecayConfig(model_type=decay_type, **decay_params),
                    experiment=ExperimentConfig(
                        max_timesteps=500, metrics_interval=50
                    ),
                    environment_name="simple_room",
                )

                size_label = f"{int(w/res)}x{int(h/res)}"
                runs.append(ExperimentRun(
                    name=f"exp5_{size_label}_{decay_type.value}_s{seed}",
                    config=cfg,
                    seed=seed,
                    tags={
                        "experiment": 5,
                        "grid_size": size_label,
                        "width_m": w,
                        "height_m": h,
                        "decay_model": decay_type.value,
                        "seed": seed,
                    }
                ))

    return runs


# Registry
EXPERIMENT_REGISTRY = {
    1: ("Decay Model Comparison", get_experiment_1_runs),
    2: ("Decay Rate Sensitivity", get_experiment_2_runs),
    3: ("Navigation Strategy Ablation", get_experiment_3_runs),
    4: ("Dynamic Environment Adaptation", get_experiment_4_runs),
    5: ("Scalability Analysis", get_experiment_5_runs),
}


def get_experiment_runs(experiment_id: int) -> List[ExperimentRun]:
    """Get all runs for a given experiment ID (1-5)."""
    if experiment_id not in EXPERIMENT_REGISTRY:
        raise ValueError(
            f"Unknown experiment {experiment_id}. "
            f"Available: {list(EXPERIMENT_REGISTRY.keys())}"
        )
    _, factory = EXPERIMENT_REGISTRY[experiment_id]
    return factory()


def list_experiments() -> None:
    """Print summary of all experiments."""
    for eid, (name, factory) in EXPERIMENT_REGISTRY.items():
        runs = factory()
        print(f"  Experiment {eid}: {name} ({len(runs)} runs)")
