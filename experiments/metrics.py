"""
Metrics collector — computes all evaluation metrics for the simulation.

Tracks per-timestep measurements and provides final summary statistics.
All metrics are designed for comparison across decay models and environments.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from slam.occupancy_grid import DecayingOccupancyGrid
from environments.grid_world import GridWorld

# Optional SSIM import
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False


@dataclass
class TimestepMetrics:
    """Metrics recorded at a single timestep."""
    timestep: int = 0
    map_mse: float = 0.0
    map_ssim: float = 0.0
    coverage: float = 0.0
    path_efficiency: float = 0.0
    total_distance: float = 0.0
    reexploration_ratio: float = 0.0
    memory_usage: int = 0
    map_entropy: float = 0.0
    localization_rmse: float = 0.0
    decay_recovery_rate: float = 0.0
    nav_success_rate: float = 0.0
    certain_occupied: int = 0
    certain_free: int = 0
    uncertain: int = 0
    never_observed: int = 0


class MetricsCollector:
    """
    Collects and computes all evaluation metrics during a simulation run.

    Metrics computed:
        1.  Map MSE — pixel-wise error vs ground truth
        2.  Map SSIM — structural similarity vs ground truth
        3.  Coverage % — observed cells / total free cells
        4.  Path Efficiency — optimal path / actual path
        5.  Total Distance — cumulative odometry
        6.  Re-exploration Ratio — revisited / total visited
        7.  Memory Usage — cells with |log_odds| > threshold
        8.  Map Entropy — total Shannon entropy
        9.  Localization RMSE — pose estimation error
        10. Decay Recovery Rate — re-confirmed / total decayed
        11. Time to X% Coverage — timestep when coverage >= X
        12. Navigation Success Rate — successful goals / total goals
    """

    def __init__(self, ground_truth: GridWorld):
        """
        Args:
            ground_truth: The ground-truth GridWorld for comparison.
        """
        self.ground_truth = ground_truth
        self._gt_grid = ground_truth.get_ground_truth_grid().astype(np.float32)
        self._gt_free_count = ground_truth.get_free_cells_count()

        # Per-timestep history
        self.history: List[TimestepMetrics] = []

        # Running counters
        self._visited_cells = set()     # (row, col) cells ever visited
        self._revisited_count = 0       # Total revisit events
        self._visit_count = 0           # Total visit events
        self._goals_attempted = 0
        self._goals_reached = 0
        self._decayed_cells_total = 0
        self._decayed_cells_recovered = 0

        # Coverage milestones
        self._coverage_milestones = {25: None, 50: None, 75: None, 90: None, 95: None}

        # Localization error tracking
        self._loc_errors_sq = []

    # ------------------------------------------------------------------
    # Per-timestep recording
    # ------------------------------------------------------------------

    def record(self, timestep: int,
               grid: DecayingOccupancyGrid,
               true_pose: Tuple[float, float, float],
               est_pose: Tuple[float, float, float],
               total_distance: float,
               scan_cells: Optional[List[Tuple[int, int]]] = None):
        """
        Record metrics at a given timestep.

        Args:
            timestep: Current simulation step.
            grid: Current occupancy grid state.
            true_pose: Ground truth robot pose.
            est_pose: Estimated robot pose.
            total_distance: Cumulative distance traveled.
            scan_cells: Optional list of cells observed this step.
        """
        m = TimestepMetrics(timestep=timestep)

        # --- Map MSE ---
        prob_map = grid.get_probability_map()
        # Resize if needed (ground truth and grid may differ in shape)
        if prob_map.shape == self._gt_grid.shape:
            m.map_mse = float(np.mean((prob_map - self._gt_grid) ** 2))
        else:
            m.map_mse = -1.0  # Shape mismatch

        # --- Map SSIM ---
        if HAS_SSIM and prob_map.shape == self._gt_grid.shape:
            try:
                m.map_ssim = float(ssim(prob_map, self._gt_grid,
                                        data_range=1.0))
            except Exception:
                m.map_ssim = -1.0
        else:
            m.map_ssim = -1.0

        # --- Coverage ---
        m.coverage = grid.get_coverage(self.ground_truth)

        # Update coverage milestones
        for pct, ts in self._coverage_milestones.items():
            if ts is None and m.coverage >= pct / 100.0:
                self._coverage_milestones[pct] = timestep

        # --- Total Distance ---
        m.total_distance = total_distance

        # --- Re-exploration Ratio ---
        if scan_cells:
            for cell in scan_cells:
                self._visit_count += 1
                if cell in self._visited_cells:
                    self._revisited_count += 1
                self._visited_cells.add(cell)

        m.reexploration_ratio = (self._revisited_count / max(self._visit_count, 1))

        # --- Memory Usage ---
        usage = grid.get_memory_usage()
        m.memory_usage = usage["total_observed"]
        m.certain_occupied = usage["certain_occupied"]
        m.certain_free = usage["certain_free"]
        m.uncertain = usage["uncertain"]
        m.never_observed = usage["never_observed"]

        # --- Map Entropy ---
        m.map_entropy = grid.get_map_entropy()

        # --- Localization RMSE ---
        loc_err = np.sqrt((true_pose[0] - est_pose[0])**2 +
                          (true_pose[1] - est_pose[1])**2)
        self._loc_errors_sq.append(loc_err ** 2)
        m.localization_rmse = np.sqrt(np.mean(self._loc_errors_sq))

        # --- Decay Recovery Rate ---
        if self._decayed_cells_total > 0:
            m.decay_recovery_rate = (self._decayed_cells_recovered /
                                     self._decayed_cells_total)
        else:
            m.decay_recovery_rate = 1.0

        # --- Navigation Success Rate ---
        if self._goals_attempted > 0:
            m.nav_success_rate = self._goals_reached / self._goals_attempted
        else:
            m.nav_success_rate = 1.0

        # --- Path Efficiency ---
        if total_distance > 0 and m.coverage > 0.01:
            # Ideal distance: straight-line traversal of observed area
            observed_area = m.memory_usage * (grid.resolution ** 2)
            ideal_dist = np.sqrt(observed_area)  # Rough approximation
            m.path_efficiency = min(ideal_dist / total_distance, 1.0)
        else:
            m.path_efficiency = 0.0

        self.history.append(m)

    # ------------------------------------------------------------------
    # Event tracking
    # ------------------------------------------------------------------

    def record_goal_attempt(self, reached: bool):
        """Record a navigation goal attempt."""
        self._goals_attempted += 1
        if reached:
            self._goals_reached += 1

    def record_decay_event(self, cells_decayed: int, cells_recovered: int):
        """Record decay and recovery events."""
        self._decayed_cells_total += cells_decayed
        self._decayed_cells_recovered += cells_recovered

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def get_final_metrics(self) -> Dict:
        """
        Return summary statistics from the entire run.

        Returns dict with final values and statistics.
        """
        if not self.history:
            return {}

        final = self.history[-1]

        return {
            # Final snapshot
            "final_map_mse": final.map_mse,
            "final_map_ssim": final.map_ssim,
            "final_coverage": final.coverage,
            "final_path_efficiency": final.path_efficiency,
            "total_distance": final.total_distance,
            "final_reexploration_ratio": final.reexploration_ratio,
            "final_memory_usage": final.memory_usage,
            "final_map_entropy": final.map_entropy,
            "final_localization_rmse": final.localization_rmse,
            "final_decay_recovery_rate": final.decay_recovery_rate,
            "final_nav_success_rate": final.nav_success_rate,
            # Coverage milestones
            "time_to_25_coverage": self._coverage_milestones.get(25),
            "time_to_50_coverage": self._coverage_milestones.get(50),
            "time_to_75_coverage": self._coverage_milestones.get(75),
            "time_to_90_coverage": self._coverage_milestones.get(90),
            "time_to_95_coverage": self._coverage_milestones.get(95),
            # Counts
            "total_timesteps": len(self.history),
            "goals_attempted": self._goals_attempted,
            "goals_reached": self._goals_reached,
            "unique_cells_visited": len(self._visited_cells),
            "total_visit_events": self._visit_count,
            "total_revisit_events": self._revisited_count,
        }

    def get_history_as_dict(self) -> Dict[str, List]:
        """
        Return per-timestep history as a dict of lists (for DataFrame).
        """
        if not self.history:
            return {}

        keys = [
            "timestep", "map_mse", "map_ssim", "coverage",
            "path_efficiency", "total_distance", "reexploration_ratio",
            "memory_usage", "map_entropy", "localization_rmse",
            "decay_recovery_rate", "nav_success_rate",
            "certain_occupied", "certain_free", "uncertain", "never_observed",
        ]

        result = {k: [] for k in keys}
        for m in self.history:
            for k in keys:
                result[k].append(getattr(m, k))

        return result

    def reset(self):
        """Reset all metrics for a new run."""
        self.history.clear()
        self._visited_cells.clear()
        self._revisited_count = 0
        self._visit_count = 0
        self._goals_attempted = 0
        self._goals_reached = 0
        self._decayed_cells_total = 0
        self._decayed_cells_recovered = 0
        self._coverage_milestones = {25: None, 50: None, 75: None, 90: None, 95: None}
        self._loc_errors_sq.clear()

    def __repr__(self) -> str:
        n = len(self.history)
        if n == 0:
            return "MetricsCollector(empty)"
        final = self.history[-1]
        return (f"MetricsCollector(steps={n}, "
                f"coverage={final.coverage:.1%}, "
                f"mse={final.map_mse:.4f})")
