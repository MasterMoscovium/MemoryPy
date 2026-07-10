#!/usr/bin/env python3
"""
MemoryPy — Run full experiment suites.

Usage:
    python run_experiments.py --experiment 1                  # Run Exp 1
    python run_experiments.py --experiment 2 --dry-run        # Preview Exp 2
    python run_experiments.py --experiment 1 --max-runs 5     # Run first 5 only
    python run_experiments.py --list                          # List all experiments
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))

from experiments.experiment_configs import (
    get_experiment_runs, list_experiments, EXPERIMENT_REGISTRY
)
from experiments.runner import ExperimentExecutor


def parse_args():
    parser = argparse.ArgumentParser(
        description="MemoryPy — Experiment Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--experiment", type=int, default=None,
                        choices=[1, 2, 3, 4, 5],
                        help="Experiment ID to run (1-5)")
    parser.add_argument("--list", action="store_true",
                        help="List all available experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview runs without executing")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="Limit number of runs (for testing)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress bars")

    return parser.parse_args()


def main():
    args = parse_args()

    # List mode
    if args.list:
        print("\n📋 Available Experiments:")
        print("=" * 50)
        list_experiments()
        print("=" * 50)
        print("\nUsage: python run_experiments.py --experiment <ID>")
        return

    if args.experiment is None:
        print("Error: --experiment ID is required. Use --list to see options.")
        sys.exit(1)

    # Get runs
    runs = get_experiment_runs(args.experiment)
    exp_name = EXPERIMENT_REGISTRY[args.experiment][0]

    if args.max_runs:
        runs = runs[:args.max_runs]

    print()
    print("=" * 60)
    print(f"  Experiment {args.experiment}: {exp_name}")
    print("=" * 60)
    print(f"  Total runs: {len(runs)}")
    print(f"  Output dir: {args.output_dir}")

    # Dry run
    if args.dry_run:
        print(f"\n  📝 DRY RUN — previewing configurations:\n")
        for i, run in enumerate(runs):
            tags_str = ", ".join(f"{k}={v}" for k, v in run.tags.items()
                                if k != "experiment")
            print(f"    [{i+1:3d}] {run.name}")
            print(f"          {tags_str}")
        print(f"\n  Total: {len(runs)} runs")
        print(f"  To execute: remove --dry-run flag")
        return

    # Execute
    print()
    start = time.time()
    executor = ExperimentExecutor(results_dir=args.output_dir)
    summary_df = executor.execute_runs(
        runs, args.experiment, progress=not args.quiet
    )
    elapsed = time.time() - start

    # Print summary
    if not summary_df.empty:
        print(f"\n{'=' * 60}")
        print(f"  EXPERIMENT {args.experiment} COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Runs completed:  {len(summary_df)}/{len(runs)}")
        print(f"  Total time:      {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"  Avg time/run:    {elapsed/max(len(summary_df),1):.1f}s")

        # Key metric summaries
        if "final_coverage" in summary_df.columns:
            print(f"  Avg coverage:    {summary_df['final_coverage'].mean():.1%}")
        if "final_map_mse" in summary_df.columns:
            print(f"  Avg map MSE:     {summary_df['final_map_mse'].mean():.4f}")
        if "total_distance" in summary_df.columns:
            print(f"  Avg distance:    {summary_df['total_distance'].mean():.1f}m")

        print(f"{'=' * 60}")
    else:
        print("\n❌ No runs completed successfully.")


if __name__ == "__main__":
    main()
