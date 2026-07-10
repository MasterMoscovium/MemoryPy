"""
Generate publication-quality figures from experiment results.

Reads summary CSVs and per-run history files, produces
high-DPI PNGs and vector PDFs for the research paper.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from visualization.plot_results import (
    setup_style,
    plot_coverage_curves,
    plot_accuracy_over_time,
    plot_pareto_front,
    plot_sensitivity,
    plot_heatmap_grid,
    plot_box_plots,
    plot_radar_chart,
    plot_bar_chart,
    MODEL_NAMES, MODEL_COLORS,
)

import matplotlib.pyplot as plt


def generate_decay_curve_comparison(output_dir: str):
    """
    Generate theoretical decay curve comparison figure.

    Shows R(Δt) for each decay model on a single plot.
    """
    setup_style()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from memory.decay_models import (
        NoDecay, ExponentialDecay, PowerLawDecay,
        AdaptiveDecay, AggressiveDecay, ThresholdDecay,
    )

    dt = np.linspace(0, 200, 500)
    visit_count = 5  # Fixed visit count for comparison

    models = [
        ("none", NoDecay()),
        ("exponential", ExponentialDecay(decay_lambda=0.01)),
        ("power_law", PowerLawDecay(beta=0.5)),
        ("adaptive", AdaptiveDecay(decay_lambda=0.01, alpha=0.3)),
        ("aggressive", AggressiveDecay(decay_lambda=0.2)),
        ("threshold", ThresholdDecay(time_threshold=100.0)),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_key, model in models:
        R = model.compute_retention(dt, np.full_like(dt, visit_count, dtype=int))
        color = MODEL_COLORS.get(model_key, "#888")
        label = MODEL_NAMES.get(model_key, model_key)
        ax.plot(dt, R, color=color, label=label, linewidth=2)

    ax.set_xlabel("Time Since Last Observation (Δt)", fontsize=12)
    ax.set_ylabel("Retention Factor R(Δt)", fontsize=12)
    ax.set_title("Memory Decay Model Comparison (visit_count=5)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=10, loc="center right")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)

    # Save both PNG and PDF
    fig.savefig(os.path.join(output_dir, "decay_curves.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "decay_curves.pdf"))
    plt.close(fig)
    print(f"📊 Saved: decay_curves.png + .pdf")


def generate_adaptive_stability_figure(output_dir: str):
    """
    Show how adaptive decay R changes with visit count.
    """
    setup_style()

    from memory.decay_models import AdaptiveDecay

    dt = np.linspace(0, 200, 500)
    visit_counts = [1, 5, 10, 20, 50]

    model = AdaptiveDecay(decay_lambda=0.01, alpha=0.3)

    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.cm.viridis
    for i, vc in enumerate(visit_counts):
        R = model.compute_retention(dt, np.full_like(dt, vc, dtype=int))
        color = cmap(i / len(visit_counts))
        ax.plot(dt, R, color=color, label=f"visits={vc}", linewidth=2)

    ax.set_xlabel("Time Since Last Observation (Δt)")
    ax.set_ylabel("Retention Factor R(Δt)")
    ax.set_title("Adaptive Decay — Effect of Visit Frequency",
                 fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "adaptive_stability.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "adaptive_stability.pdf"))
    plt.close(fig)
    print(f"📊 Saved: adaptive_stability.png + .pdf")


def generate_all_paper_figures(results_dir: str = "results"):
    """Generate all figures needed for the paper."""
    summary_dir = os.path.join(results_dir, "summary")
    output_dir = os.path.join(results_dir, "figures", "paper")

    print()
    print("=" * 60)
    print("  Generating Paper Figures")
    print("=" * 60)

    # Theoretical figures (always available, no data needed)
    generate_decay_curve_comparison(output_dir)
    generate_adaptive_stability_figure(output_dir)

    # Data-dependent figures
    exp1_summary = os.path.join(summary_dir, "experiment_1_summary.csv")
    exp2_summary = os.path.join(summary_dir, "experiment_2_summary.csv")

    if os.path.exists(exp1_summary):
        plot_coverage_curves(summary_dir, output_dir)
        plot_accuracy_over_time(summary_dir, output_dir)
        plot_pareto_front(exp1_summary, output_dir)
        plot_heatmap_grid(exp1_summary, output_dir)
        plot_box_plots(exp1_summary, output_dir)
        plot_radar_chart(exp1_summary, output_dir)
        plot_bar_chart(exp1_summary, output_dir)
    else:
        print(f"⚠️  No Experiment 1 results at {exp1_summary}")
        print("   Run: python run_experiments.py --experiment 1")

    if os.path.exists(exp2_summary):
        plot_sensitivity(exp2_summary, output_dir)
    else:
        print(f"⚠️  No Experiment 2 results at {exp2_summary}")

    print(f"\n📂 Paper figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_paper_figures()
