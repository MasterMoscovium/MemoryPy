"""
Publication-quality plotting — 8 plot types for research paper figures.

All plots use a consistent style with seaborn + matplotlib.
Outputs publication-ready figures at 300 DPI.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import os


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

def setup_style():
    """Configure global matplotlib/seaborn style for publication."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "grid.alpha": 0.3,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
    })


# Model display names and colors
MODEL_NAMES = {
    "none": "No Decay",
    "exponential": "Exponential",
    "power_law": "Power-Law",
    "adaptive": "Adaptive",
    "aggressive": "Aggressive",
    "threshold": "Threshold",
}

MODEL_COLORS = {
    "none": "#7f8c8d",
    "exponential": "#3498db",
    "power_law": "#e67e22",
    "adaptive": "#2ecc71",
    "aggressive": "#e74c3c",
    "threshold": "#9b59b6",
}


# ---------------------------------------------------------------------------
# Plot 1: Coverage Curves Over Time
# ---------------------------------------------------------------------------

def plot_coverage_curves(summary_dir: str, output_dir: str,
                         experiment_id: int = 1):
    """
    Plot coverage vs. timestep for each decay model (averaged over seeds).

    One subplot per environment.
    """
    setup_style()

    # Load all history CSVs for this experiment
    raw_dir = os.path.join(os.path.dirname(summary_dir), "raw")
    if not os.path.exists(raw_dir):
        print(f"⚠️  No raw data in {raw_dir}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Coverage Over Time by Decay Model", fontsize=14, fontweight="bold")

    environments = ["simple_room", "office_maze", "open_field", "dynamic_obstacles"]

    for idx, (ax, env) in enumerate(zip(axes.flat, environments)):
        for model_name, color in MODEL_COLORS.items():
            # Find matching history files
            pattern = f"exp1_{env}_{model_name}_"
            histories = []

            for f in os.listdir(raw_dir):
                if f.startswith(pattern) and f.endswith("_history.csv"):
                    try:
                        df = pd.read_csv(os.path.join(raw_dir, f))
                        if "coverage" in df.columns:
                            histories.append(df)
                    except Exception:
                        continue

            if not histories:
                continue

            # Average coverage across seeds
            all_cov = pd.concat([h[["timestep", "coverage"]] for h in histories])
            mean_cov = all_cov.groupby("timestep")["coverage"].agg(["mean", "std"])

            ax.plot(mean_cov.index, mean_cov["mean"],
                    color=color, label=MODEL_NAMES.get(model_name, model_name),
                    linewidth=1.5)

            if "std" in mean_cov.columns and len(histories) > 1:
                ax.fill_between(mean_cov.index,
                                mean_cov["mean"] - mean_cov["std"],
                                mean_cov["mean"] + mean_cov["std"],
                                alpha=0.15, color=color)

        ax.set_title(env.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Coverage")
        ax.set_ylim(0, 1.05)

        if idx >= 2:
            ax.set_xlabel("Timestep")

    axes[0, 0].legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "coverage_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"📊 Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2: Map Accuracy Over Time
# ---------------------------------------------------------------------------

def plot_accuracy_over_time(summary_dir: str, output_dir: str):
    """Plot MSE and SSIM over time for each decay model."""
    setup_style()

    raw_dir = os.path.join(os.path.dirname(summary_dir), "raw")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Map Accuracy Over Time (Office Maze)", fontsize=14, fontweight="bold")

    for model_name, color in MODEL_COLORS.items():
        pattern = f"exp1_office_maze_{model_name}_"
        histories = []

        for f in os.listdir(raw_dir) if os.path.exists(raw_dir) else []:
            if f.startswith(pattern) and f.endswith("_history.csv"):
                try:
                    histories.append(pd.read_csv(os.path.join(raw_dir, f)))
                except Exception:
                    continue

        if not histories:
            continue

        all_data = pd.concat(histories)
        mean_data = all_data.groupby("timestep").mean(numeric_only=True)

        label = MODEL_NAMES.get(model_name, model_name)
        if "map_mse" in mean_data.columns:
            ax1.plot(mean_data.index, mean_data["map_mse"],
                     color=color, label=label, linewidth=1.5)
        if "map_ssim" in mean_data.columns:
            valid = mean_data["map_ssim"] > -0.5
            ax2.plot(mean_data.index[valid], mean_data["map_ssim"][valid],
                     color=color, label=label, linewidth=1.5)

    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Map MSE (lower is better)")
    ax1.set_title("Mean Squared Error")
    if ax1.get_legend_handles_labels()[1]:
        ax1.legend(fontsize=8)

    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("SSIM (higher is better)")
    ax2.set_title("Structural Similarity")
    if ax2.get_legend_handles_labels()[1]:
        ax2.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "accuracy_over_time.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"📊 Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3: Pareto Front — Coverage vs Memory Usage
# ---------------------------------------------------------------------------

def plot_pareto_front(summary_path: str, output_dir: str):
    """Plot coverage vs memory usage Pareto front."""
    setup_style()

    if not os.path.exists(summary_path):
        print(f"⚠️  No summary at {summary_path}")
        return

    df = pd.read_csv(summary_path)
    if "final_coverage" not in df.columns or "final_memory_usage" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in df["decay_model"].unique():
        subset = df[df["decay_model"] == model]
        color = MODEL_COLORS.get(model, "#888888")
        label = MODEL_NAMES.get(model, model)
        ax.scatter(subset["final_memory_usage"], subset["final_coverage"],
                   c=color, label=label, s=60, alpha=0.7, edgecolors="white", linewidth=0.5)

    ax.set_xlabel("Memory Usage (observed cells)")
    ax.set_ylabel("Final Coverage")
    ax.set_title("Coverage vs Memory Usage — Pareto Front", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "pareto_front.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"📊 Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 4: Sensitivity Analysis (λ sweep)
# ---------------------------------------------------------------------------

def plot_sensitivity(summary_path: str, output_dir: str):
    """Plot metrics vs decay rate λ."""
    setup_style()

    if not os.path.exists(summary_path):
        return

    df = pd.read_csv(summary_path)
    if "decay_lambda" not in df.columns:
        return

    metrics = ["final_coverage", "final_map_mse", "final_map_entropy",
               "total_distance"]
    titles = ["Coverage", "Map MSE", "Map Entropy", "Distance Traveled"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Decay Rate (λ) Sensitivity Analysis", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(axes.flat, metrics, titles):
        if metric in df.columns:
            grouped = df.groupby("decay_lambda")[metric].agg(["mean", "std"])
            ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"],
                        marker='o', capsize=4, color=MODEL_COLORS["exponential"],
                        linewidth=1.5)
            ax.set_xscale("log")
            ax.set_xlabel("λ (decay rate)")
            ax.set_ylabel(title)
            ax.set_title(title)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "sensitivity_lambda.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"📊 Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 5: Heatmap Grid — Metrics × Models × Environments
# ---------------------------------------------------------------------------

def plot_heatmap_grid(summary_path: str, output_dir: str):
    """Heatmap showing metric values across decay models and environments."""
    setup_style()

    if not os.path.exists(summary_path):
        return

    df = pd.read_csv(summary_path)
    required = ["decay_model", "environment", "final_coverage"]
    if not all(c in df.columns for c in required):
        return

    metrics = ["final_coverage", "final_map_mse", "final_map_entropy"]
    titles = ["Coverage", "Map MSE", "Map Entropy"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    fig.suptitle("Performance Heatmap", fontsize=14, fontweight="bold")

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric, title in zip(axes, metrics, titles):
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(values=metric, index="decay_model",
                               columns="environment", aggfunc="mean")
        # Rename for display
        pivot.index = [MODEL_NAMES.get(m, m) for m in pivot.index]

        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                    ax=ax, linewidths=0.5)
        ax.set_title(title)
        ax.set_ylabel("Decay Model")
        ax.set_xlabel("Environment")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "heatmap_grid.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"📊 Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 6: Box Plots — Metric Distribution by Model
# ---------------------------------------------------------------------------

def plot_box_plots(summary_path: str, output_dir: str):
    """Box plots of key metrics grouped by decay model."""
    setup_style()

    if not os.path.exists(summary_path):
        return

    df = pd.read_csv(summary_path)
    if "decay_model" not in df.columns:
        return

    # Map names for display
    df["Model"] = df["decay_model"].map(lambda x: MODEL_NAMES.get(x, x))

    metrics = ["final_coverage", "final_map_mse", "final_localization_rmse",
               "final_map_entropy"]
    titles = ["Final Coverage", "Map MSE", "Localization RMSE", "Map Entropy"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Metric Distributions by Decay Model", fontsize=14, fontweight="bold")

    palette = [MODEL_COLORS.get(m, "#888") for m in df["decay_model"].unique()]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        if metric in df.columns:
            sns.boxplot(data=df, x="Model", y=metric, hue="Model", ax=ax,
                        palette=dict(zip(df["Model"].unique(), palette)),
                        linewidth=0.8, legend=False)
            ax.set_title(title)
            ax.set_xlabel("")
            ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "box_plots.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"📊 Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 7: Radar Chart — Multi-metric Model Comparison
# ---------------------------------------------------------------------------

def plot_radar_chart(summary_path: str, output_dir: str):
    """Spider/radar chart comparing normalized metrics per model."""
    setup_style()

    if not os.path.exists(summary_path):
        return

    df = pd.read_csv(summary_path)
    if "decay_model" not in df.columns:
        return

    metrics = ["final_coverage", "final_path_efficiency", "final_nav_success_rate",
               "final_decay_recovery_rate"]
    metric_labels = ["Coverage", "Path Eff.", "Nav Success", "Decay Recovery"]

    # Only use metrics that exist
    valid = [(m, l) for m, l in zip(metrics, metric_labels) if m in df.columns]
    if len(valid) < 3:
        return
    metrics, metric_labels = zip(*valid)

    # Compute means per model
    means = df.groupby("decay_model")[list(metrics)].mean()

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in means.index:
        values = means.loc[model].tolist()
        values += values[:1]
        color = MODEL_COLORS.get(model, "#888888")
        label = MODEL_NAMES.get(model, model)
        ax.plot(angles, values, 'o-', linewidth=1.5, color=color, label=label)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Multi-metric Model Comparison", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "radar_chart.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"📊 Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 8: Bar Chart — Coverage Milestones
# ---------------------------------------------------------------------------

def plot_bar_chart(summary_path: str, output_dir: str):
    """Bar chart of time-to-coverage milestones per model."""
    setup_style()

    if not os.path.exists(summary_path):
        return

    df = pd.read_csv(summary_path)
    milestones = ["time_to_25_coverage", "time_to_50_coverage",
                  "time_to_75_coverage", "time_to_90_coverage"]
    milestone_labels = ["25%", "50%", "75%", "90%"]

    valid = [m for m in milestones if m in df.columns]
    if not valid or "decay_model" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    models = df["decay_model"].unique()
    x = np.arange(len(models))
    width = 0.2

    for i, (milestone, label) in enumerate(zip(valid, milestone_labels)):
        means = []
        for model in models:
            val = df[df["decay_model"] == model][milestone].mean()
            means.append(val if not np.isnan(val) else 0)
        ax.bar(x + i * width, means, width, label=f"{label} coverage",
               alpha=0.85)

    ax.set_xlabel("Decay Model")
    ax.set_ylabel("Timesteps to Reach Coverage")
    ax.set_title("Time to Coverage Milestones", fontweight="bold")
    ax.set_xticks(x + width * len(valid) / 2)
    ax.set_xticklabels([MODEL_NAMES.get(m, m) for m in models], rotation=30)
    ax.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "bar_coverage_milestones.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"📊 Saved: {path}")


# ---------------------------------------------------------------------------
# Generate all plots
# ---------------------------------------------------------------------------

def generate_all_plots(results_dir: str = "results"):
    """Generate all 8 plot types from experiment results."""
    summary_dir = os.path.join(results_dir, "summary")
    output_dir = os.path.join(results_dir, "figures")

    print("📈 Generating publication plots...")
    print("=" * 50)

    # Experiment 1 plots
    exp1_summary = os.path.join(summary_dir, "experiment_1_summary.csv")
    plot_coverage_curves(summary_dir, output_dir)
    plot_accuracy_over_time(summary_dir, output_dir)
    plot_pareto_front(exp1_summary, output_dir)
    plot_heatmap_grid(exp1_summary, output_dir)
    plot_box_plots(exp1_summary, output_dir)
    plot_radar_chart(exp1_summary, output_dir)
    plot_bar_chart(exp1_summary, output_dir)

    # Experiment 2 plots
    exp2_summary = os.path.join(summary_dir, "experiment_2_summary.csv")
    plot_sensitivity(exp2_summary, output_dir)

    print("=" * 50)
    print(f"📂 All figures saved to: {output_dir}")


if __name__ == "__main__":
    generate_all_plots()
