# MemoryPy — Memory-Decay Based Autonomous Robot Navigation Using SLAM

A Python simulation framework for experimentally analyzing how **biologically-inspired memory decay** (Ebbinghaus forgetting curve) affects autonomous robot navigation when integrated into a SLAM pipeline.

## Research Hypothesis

- **H1**: Incorporating memory decay into SLAM-based navigation improves long-term map relevance and reduces memory footprint, with an optimal decay rate balancing coverage, accuracy, and resources.
- **H2**: Adaptive decay models (visit-frequency-dependent) outperform fixed-rate decay in both static and dynamic environments.

## Architecture

```
MemoryPy/
├── config/          # Simulation parameters (dataclasses)
├── environments/    # 2D grid worlds + JSON map definitions
├── robot/           # Differential-drive kinematics + LiDAR sensor
├── slam/            # Occupancy grid, particle filter, scan matching
├── memory/          # Decay models (exponential, power-law, adaptive)
├── navigation/      # Frontier exploration, A* planning, goal selection
├── experiments/     # Experiment configs, runner, metrics
├── visualization/   # Live dashboard, plots, video recording
├── paper/           # Figure + LaTeX table generation
└── results/         # Experiment output (auto-generated)
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single simulation with visualization
python main.py --env simple_room --decay exponential --steps 1000 --visualize

# Run the full experiment suite
python run_experiments.py --experiment 1 --parallel 4
```

## Environments

| Environment | Size | Description |
|---|---|---|
| `simple_room` | 20×20m | Single room with scattered obstacles |
| `office_maze` | 40×30m | Multi-room layout with corridors |
| `open_field` | 50×50m | Large open area, sparse obstacles |
| `dynamic_obstacles` | 30×30m | Objects change position over time |

## Decay Models

1. **Exponential** — `R = e^(-λΔt)` (Ebbinghaus forgetting curve)
2. **Power-Law** — `R = (Δt+1)^(-β)` (Jost's Law)
3. **Adaptive** — `R = e^(-λ/S·Δt)` where `S` grows with visit count
4. **No Decay** — Control baseline (standard SLAM)
5. **Threshold** — Binary forget below certainty threshold

## Experiments

1. **Decay Model Comparison** — 6 models × 4 environments × 5 seeds
2. **Decay Rate Sensitivity** — λ sweep for exponential model
3. **Navigation Strategy Ablation** — 5 goal-selection weight presets
4. **Dynamic Environment Adaptation** — Decay vs. environmental change
5. **Scalability Analysis** — Computational cost vs. grid size

## License

Research use. See implementation plan for full methodology.
