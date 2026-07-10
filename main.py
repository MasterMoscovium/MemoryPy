#!/usr/bin/env python3
"""
MemoryPy — Single simulation entry point.

Run a single simulation with visualization or headless mode.

Usage:
    python main.py --env simple_room --decay exponential --steps 500
    python main.py --env office_maze --decay adaptive --steps 1000 --visualize
    python main.py --help
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config.settings import (
    SimulationConfig, DecayConfig, DecayModelType,
    NavigationConfig, NavigationStrategy
)
from experiments.runner import SimulationRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="MemoryPy — Memory-Decay SLAM Robot Navigation Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --env simple_room --decay none --steps 200
  python main.py --env office_maze --decay exponential --lambda 0.01 --steps 1000
  python main.py --env dynamic_obstacles --decay adaptive --steps 2000 --seed 42
        """
    )

    parser.add_argument("--env", type=str, default="simple_room",
                        choices=["simple_room", "office_maze",
                                 "open_field", "dynamic_obstacles"],
                        help="Environment map name (default: simple_room)")
    parser.add_argument("--decay", type=str, default="exponential",
                        choices=["none", "exponential", "power_law",
                                 "adaptive", "aggressive", "threshold"],
                        help="Decay model (default: exponential)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of simulation steps (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--particles", type=int, default=15,
                        help="Number of SLAM particles (default: 15)")

    # Decay parameters
    parser.add_argument("--lambda", dest="decay_lambda", type=float,
                        default=0.01, help="Decay rate λ (default: 0.01)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Power-law β (default: 0.5)")

    # Navigation
    parser.add_argument("--strategy", type=str, default="balanced",
                        choices=["exploration_heavy", "balanced",
                                 "decay_priority", "greedy", "random"],
                        help="Navigation strategy (default: balanced)")

    # Output
    parser.add_argument("--visualize", action="store_true",
                        help="Show live pygame visualization dashboard")
    parser.add_argument("--record", type=str, default=None,
                        help="Path to save video recording (.mp4 or .gif)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress bar")

    return parser.parse_args()


def main():
    args = parse_args()

    # Build configuration
    cfg = SimulationConfig.for_environment(args.env)

    # Decay config
    decay_type = DecayModelType(args.decay)
    cfg.decay = DecayConfig(
        model_type=decay_type,
        decay_lambda=args.decay_lambda,
        power_beta=args.beta,
    )

    # Particle filter
    cfg.particle_filter.num_particles = args.particles

    # Navigation
    cfg.navigation.strategy = NavigationStrategy(args.strategy)

    # Experiment
    cfg.experiment.max_timesteps = args.steps
    cfg.experiment.metrics_interval = max(args.steps // 40, 1)

    # Dynamic obstacles
    cfg.enable_dynamic_obstacles = (args.env == "dynamic_obstacles")

    # Print configuration
    print("=" * 60)
    print("  MemoryPy — Memory-Decay SLAM Navigation")
    print("=" * 60)
    print(f"  Environment:  {args.env}")
    print(f"  Decay Model:  {decay_type.value} (λ={args.decay_lambda})")
    print(f"  Strategy:     {args.strategy}")
    print(f"  Steps:        {args.steps}")
    print(f"  Particles:    {args.particles}")
    print(f"  Seed:         {args.seed}")
    print(f"  Grid Size:    {cfg.grid.shape}")
    if args.visualize:
        print("  Visualizing:  Live Dashboard")
    if args.record:
        print(f"  Recording:    {args.record}")
    print("=" * 60)
    print()

    # Run simulation
    runner = SimulationRunner(cfg, seed=args.seed)
    results = runner.run(
        progress=not args.quiet,
        visualize=args.visualize,
        video_path=args.record
    )

    # Print results
    print()
    print("═" * 60)
    print("  RESULTS")
    print("═" * 60)
    print(f"  Coverage:          {results.get('final_coverage', 0):.1%}")
    print(f"  Map MSE:           {results.get('final_map_mse', 0):.4f}")
    print(f"  Map SSIM:          {results.get('final_map_ssim', -1):.4f}")
    print(f"  Localization RMSE: {results.get('final_localization_rmse', 0):.4f}m")
    print(f"  Total Distance:    {results.get('total_distance', 0):.2f}m")
    print(f"  Path Efficiency:   {results.get('final_path_efficiency', 0):.3f}")
    print(f"  Memory Usage:      {results.get('final_memory_usage', 0)} cells")
    print(f"  Map Entropy:       {results.get('final_map_entropy', 0):.1f}")
    print(f"  Re-explore Ratio:  {results.get('final_reexploration_ratio', 0):.3f}")
    print(f"  Nav Success Rate:  {results.get('final_nav_success_rate', 0):.1%}")
    print(f"  Run Time:          {results.get('run_time_seconds', 0):.1f}s")

    t50 = results.get('time_to_50_coverage')
    t90 = results.get('time_to_90_coverage')
    if t50 is not None:
        print(f"  Time to 50% cov:   step {t50}")
    if t90 is not None:
        print(f"  Time to 90% cov:   step {t90}")

    print("═" * 60)

    # Save history CSV
    history_df = runner.get_history_dataframe()
    if not history_df.empty:
        os.makedirs(os.path.join(args.output_dir, "raw"), exist_ok=True)
        csv_path = os.path.join(
            args.output_dir, "raw",
            f"single_{args.env}_{args.decay}_s{args.seed}.csv"
        )
        history_df.to_csv(csv_path, index=False)
        print(f"\n📊 History saved: {csv_path}")


if __name__ == "__main__":
    main()
