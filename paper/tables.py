"""
LaTeX table generation — produces paper-ready tables from experiment results.

Generates:
    1. Parameter table — simulation configuration
    2. Metric comparison table — per-model summary
    3. Statistical significance table — pairwise tests
    4. Computational cost table — scalability results
"""

import os
import numpy as np
import pandas as pd
from typing import Optional


def generate_parameter_table() -> str:
    """
    Generate LaTeX table of simulation parameters.
    """
    rows = [
        ("Grid resolution", "0.1", "m/cell"),
        ("LiDAR max range", "8.0", "m"),
        ("LiDAR beams", "180", ""),
        ("LiDAR FOV", "270", "degrees"),
        ("LiDAR noise σ", "0.02", "m"),
        ("Robot max velocity", "0.5", "m/s"),
        ("Robot max angular vel.", "1.0", "rad/s"),
        ("Simulation dt", "0.1", "s"),
        ("Odometry noise α₁", "0.05", ""),
        ("Odometry noise α₃", "0.05", ""),
        ("Particles (FastSLAM)", "15", ""),
        ("Decay interval", "10", "steps"),
        ("Exp. decay λ (default)", "0.01", ""),
        ("Power-law β", "0.5", ""),
        ("Adaptive α", "0.3", ""),
        ("Max timesteps", "2000", ""),
        ("Seeds per config", "5", ""),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Simulation Parameters}",
        r"\label{tab:parameters}",
        r"\begin{tabular}{lrc}",
        r"\toprule",
        r"\textbf{Parameter} & \textbf{Value} & \textbf{Unit} \\",
        r"\midrule",
    ]

    for param, value, unit in rows:
        unit_str = unit if unit else "—"
        lines.append(f"  {param} & {value} & {unit_str} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_metric_comparison_table(summary_path: str) -> str:
    """
    Generate LaTeX table comparing metrics across decay models.
    """
    if not os.path.exists(summary_path):
        return f"% No data at {summary_path}"

    df = pd.read_csv(summary_path)
    if "decay_model" not in df.columns:
        return "% Missing decay_model column"

    model_names = {
        "none": "No Decay",
        "exponential": "Exponential",
        "power_law": "Power-Law",
        "adaptive": "Adaptive",
        "aggressive": "Aggressive",
        "threshold": "Threshold",
    }

    metrics = [
        ("final_coverage", "Coverage", ".1%", True),
        ("final_map_mse", "Map MSE", ".4f", False),
        ("final_localization_rmse", "Loc. RMSE (m)", ".4f", False),
        ("total_distance", "Distance (m)", ".1f", None),
        ("final_map_entropy", "Entropy", ".1f", False),
        ("final_reexploration_ratio", "Re-explore", ".3f", None),
    ]

    # Filter to metrics that exist
    valid_metrics = [(col, name, fmt, higher)
                     for col, name, fmt, higher in metrics
                     if col in df.columns]

    n_cols = len(valid_metrics)
    col_spec = "l" + "r" * n_cols

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Metric Comparison Across Decay Models (mean ± std, 5 seeds)}",
        r"\label{tab:metrics}",
        r"\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header row
    header = r"\textbf{Model}"
    for _, name, _, _ in valid_metrics:
        header += f" & \\textbf{{{name}}}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    for model in df["decay_model"].unique():
        subset = df[df["decay_model"] == model]
        display_name = model_names.get(model, model)
        row = f"  {display_name}"

        for col, _, fmt, higher_better in valid_metrics:
            mean = subset[col].mean()
            std = subset[col].std()

            if fmt == ".1%":
                val_str = f"{mean:.1%}"
                std_str = f"{std:.1%}"
            else:
                val_str = f"{mean:{fmt}}"
                std_str = f"{std:{fmt}}"

            row += f" & {val_str} ± {std_str}"

        row += r" \\"
        lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def generate_significance_table(summary_path: str) -> str:
    """
    Generate pairwise Mann-Whitney U test results for coverage.
    """
    if not os.path.exists(summary_path):
        return f"% No data at {summary_path}"

    df = pd.read_csv(summary_path)
    if "decay_model" not in df.columns or "final_coverage" not in df.columns:
        return "% Missing required columns"

    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return "% scipy required for significance tests"

    models = sorted(df["decay_model"].unique())
    model_names = {
        "none": "None", "exponential": "Exp.",
        "power_law": "P-L", "adaptive": "Adpt.",
        "aggressive": "Aggr.", "threshold": "Thr.",
    }

    n = len(models)
    col_spec = "l" + "c" * n

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Pairwise Significance (Mann-Whitney U, Coverage)}",
        r"\label{tab:significance}",
        r"\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header = ""
    for m in models:
        header += f" & \\textbf{{{model_names.get(m, m)}}}"
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for i, m1 in enumerate(models):
        row = f"\\textbf{{{model_names.get(m1, m1)}}}"
        d1 = df[df["decay_model"] == m1]["final_coverage"]

        for j, m2 in enumerate(models):
            if i == j:
                row += " & —"
            elif i < j:
                d2 = df[df["decay_model"] == m2]["final_coverage"]
                try:
                    _, p = mannwhitneyu(d1, d2, alternative='two-sided')
                    if p < 0.001:
                        row += " & ***"
                    elif p < 0.01:
                        row += " & **"
                    elif p < 0.05:
                        row += " & *"
                    else:
                        row += " & n.s."
                except Exception:
                    row += " & —"
            else:
                row += " &"

        row += r" \\"
        lines.append(f"  {row}")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2mm}",
        r"\footnotesize{* $p<0.05$, ** $p<0.01$, *** $p<0.001$, n.s. = not significant}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_scalability_table(summary_path: str) -> str:
    """
    Generate table of computation time vs grid size.
    """
    if not os.path.exists(summary_path):
        return f"% No data at {summary_path}"

    df = pd.read_csv(summary_path)
    required = ["grid_size", "decay_model", "run_time_seconds"]
    if not all(c in df.columns for c in required):
        return "% Missing columns for scalability table"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Computational Cost vs Grid Size}",
        r"\label{tab:scalability}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Grid Size} & \textbf{No Decay (s)} & \textbf{Exp. Decay (s)} & \textbf{Overhead (\%)} \\",
        r"\midrule",
    ]

    for size in sorted(df["grid_size"].unique()):
        subset = df[df["grid_size"] == size]
        no_decay = subset[subset["decay_model"] == "none"]["run_time_seconds"].mean()
        exp_decay = subset[subset["decay_model"] == "exponential"]["run_time_seconds"].mean()

        if no_decay > 0:
            overhead = ((exp_decay - no_decay) / no_decay) * 100
        else:
            overhead = 0

        lines.append(
            f"  {size} & {no_decay:.1f} & {exp_decay:.1f} & {overhead:.1f}\\% \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generate all tables
# ---------------------------------------------------------------------------

def generate_all_tables(results_dir: str = "results",
                        output_dir: Optional[str] = None):
    """Generate all LaTeX tables and save to files."""
    if output_dir is None:
        output_dir = os.path.join(results_dir, "figures", "paper")
    os.makedirs(output_dir, exist_ok=True)

    summary_dir = os.path.join(results_dir, "summary")

    print()
    print("=" * 60)
    print("  Generating LaTeX Tables")
    print("=" * 60)

    # 1. Parameter table (always available)
    param_table = generate_parameter_table()
    _save_table(param_table, output_dir, "table_parameters.tex")

    # 2. Metric comparison
    exp1 = os.path.join(summary_dir, "experiment_1_summary.csv")
    metric_table = generate_metric_comparison_table(exp1)
    _save_table(metric_table, output_dir, "table_metrics.tex")

    # 3. Significance tests
    sig_table = generate_significance_table(exp1)
    _save_table(sig_table, output_dir, "table_significance.tex")

    # 4. Scalability
    exp5 = os.path.join(summary_dir, "experiment_5_summary.csv")
    scale_table = generate_scalability_table(exp5)
    _save_table(scale_table, output_dir, "table_scalability.tex")

    print("=" * 60)
    print(f"📂 Tables saved to: {output_dir}")


def _save_table(content: str, output_dir: str, filename: str):
    """Save a LaTeX table to a file."""
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        f.write(content)
    print(f"  📄 {filename}")


if __name__ == "__main__":
    generate_all_tables()
