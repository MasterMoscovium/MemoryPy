"""
Experiment runner — orchestrates simulation runs, collects metrics, saves results.

Supports:
    - Single run execution with full SLAM + decay + navigation loop
    - Batch experiment execution
    - CSV output per run
    - Progress tracking
    - Reproducible seeding
"""

import os
import time
import json
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from tqdm import tqdm

from config.settings import SimulationConfig, DecayModelType
from environments.grid_world import GridWorld
from robot.differential_drive import DifferentialDriveRobot
from robot.sensors import LidarSensor
from slam.particle_filter import ParticleFilter
from memory.memory_manager import MemoryManager
from navigation.frontier_explorer import FrontierExplorer
from navigation.path_planner import AStarPlanner
from navigation.goal_selector import GoalSelector
from experiments.metrics import MetricsCollector
from experiments.experiment_configs import ExperimentRun


class SimulationRunner:
    """
    Runs a single simulation with the full pipeline:
    robot → LiDAR → SLAM → decay → navigation → metrics.
    """

    def __init__(self, config: SimulationConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Initialize components
        self.grid_world = GridWorld(
            config.environment_name,
            resolution=config.grid.resolution,
        )

        self.robot = DifferentialDriveRobot(config.robot, rng=self.rng)
        self.lidar = LidarSensor(config.lidar, rng=self.rng)

        self.particle_filter = ParticleFilter(
            config.particle_filter,
            config.robot,
            config.grid,
            config.lidar,
            initial_pose=(config.robot.start_x,
                          config.robot.start_y,
                          config.robot.start_theta),
            rng=self.rng,
        )

        self.memory_manager = MemoryManager(config.decay)

        self.frontier_explorer = FrontierExplorer(config.navigation)
        self.planner = AStarPlanner(config.navigation)
        self.goal_selector = GoalSelector(
            config.navigation, self.planner, rng=self.rng
        )

        self.metrics = MetricsCollector(self.grid_world)

        # State
        self._timestep = 0
        self._last_v = 0.0
        self._last_omega = 0.0

    def run(self, max_timesteps: Optional[int] = None,
            progress: bool = True,
            visualize: bool = False,
            video_path: Optional[str] = None) -> Dict:
        """
        Execute the full simulation loop.

        Args:
            max_timesteps: Override config max_timesteps.
            progress: Show tqdm progress bar.
            visualize: Show live pygame dashboard.
            video_path: Path to save video (.mp4 or .gif).

        Returns:
            Dict with final metrics and run metadata.
        """
        max_t = max_timesteps or self.config.experiment.max_timesteps
        interval = self.config.experiment.metrics_interval

        iterator = range(max_t)
        if progress:
            iterator = tqdm(iterator, desc="Simulating", leave=False)

        start_time = time.time()

        dashboard = None
        recorder = None
        if visualize or video_path:
            try:
                from visualization.live_dashboard import LiveDashboard
                dashboard = LiveDashboard(self.grid_world)
                if video_path:
                    from visualization.video_recorder import VideoRecorder
                    recorder = VideoRecorder(video_path, fps=15)
            except ImportError as e:
                print(f"⚠️  Could not load visualization: {e}")

        for t in iterator:
            self._timestep = t
            self._step(t)

            # Record metrics at intervals
            if t % interval == 0 or t == max_t - 1:
                grid = self.particle_filter.get_occupancy_grid()
                self.metrics.record(
                    timestep=t,
                    grid=grid,
                    true_pose=self.robot.true_pose,
                    est_pose=self.particle_filter.get_best_pose(),
                    total_distance=self.robot.total_distance,
                )

            # Visualization update
            if dashboard is not None:
                grid = self.particle_filter.get_occupancy_grid()
                frontiers = []
                reex_mask = self.memory_manager.get_reexploration_candidates(grid)
                if np.any(reex_mask) or True: # always try to get frontiers for visualization
                     frontiers = self.frontier_explorer.detect_frontiers(grid, reex_mask)
                
                live_metrics = {
                    "coverage": self.metrics.history[-1].coverage if self.metrics.history else 0.0,
                    "map_mse": self.metrics.history[-1].map_mse if self.metrics.history else 0.0,
                    "total_distance": self.robot.total_distance,
                    "model_name": self.config.decay.model_type.value
                }
                
                running = dashboard.update(
                    grid=grid,
                    true_pose=self.robot.true_pose,
                    est_pose=self.particle_filter.get_best_pose(),
                    trajectory=self.robot.est_trajectory,
                    path=self.goal_selector.current_path,
                    frontiers=frontiers,
                    metrics=live_metrics,
                    timestep=t
                )
                
                if recorder is not None:
                    recorder.add_frame(dashboard.capture_frame())
                    
                if not running:
                    print("\n🛑 Simulation stopped by user.")
                    break

        if dashboard is not None:
            dashboard.close()
        if recorder is not None:
            recorder.save()

        elapsed = time.time() - start_time

        # Compile results
        final = self.metrics.get_final_metrics()
        final["run_time_seconds"] = elapsed
        final["seed"] = self.seed
        final["environment"] = self.config.environment_name
        final["decay_model"] = self.config.decay.model_type.value

        return final

    def _step(self, t: int):
        """Execute one simulation timestep."""
        cfg = self.config

        # 1. Get current best pose estimate
        pose = self.particle_filter.get_best_pose()

        # 2. LiDAR scan from true pose (sensor sees reality)
        scan = self.lidar.scan(self.robot.true_pose, self.grid_world)

        # 3. SLAM update (particle filter)
        est_pose = self.particle_filter.update(
            self._last_v, self._last_omega, scan, float(t)
        )

        # 4. Apply memory decay
        grid = self.particle_filter.get_occupancy_grid()
        if self.memory_manager.should_apply_decay(t):
            self.memory_manager.apply_decay(grid, float(t))

        # 5. Dynamic environment updates
        if cfg.enable_dynamic_obstacles:
            self.grid_world.update(t)

        # 6. Navigation: select goal & plan path
        robot_xy = (pose[0], pose[1])
        goal_reached = self.goal_selector.is_goal_reached(robot_xy, threshold=1.0)
        needs_new_goal = (self.goal_selector.current_goal is None) or goal_reached

        if needs_new_goal or self.goal_selector.should_replan(t):
            # Record goal attempt only when goal was reached or abandoned for a new one
            if goal_reached and self.goal_selector.current_goal is not None:
                self.metrics.record_goal_attempt(True)
                self.goal_selector.clear_goal()

            reex_mask = self.memory_manager.get_reexploration_candidates(grid)
            frontiers = self.frontier_explorer.detect_frontiers(grid, reex_mask)

            old_goal = self.goal_selector.current_goal
            self.goal_selector.select_goal(
                grid, robot_xy, frontiers, reex_mask
            )

            # If we replaced an unreached goal with a new one, record as failed attempt
            if old_goal is not None and not goal_reached and self.goal_selector.current_goal != old_goal:
                self.metrics.record_goal_attempt(False)

        # 7. Compute velocity command toward next waypoint
        v, omega = self._compute_velocity(pose)

        # 8. Move robot
        v_noisy, omega_noisy = self.robot.move(
            v, omega,
            collision_fn=self.grid_world.check_collision,
        )

        # 9. Stuck detection
        # If we commanded significant forward velocity but physically didn't move (collided)
        if abs(v) > 0.1 and abs(v_noisy) < 0.01:
            # Force immediate A* replan by clearing the current goal
            self.goal_selector.current_goal = None

        self._last_v = v
        self._last_omega = omega

    def _compute_velocity(self, pose: Tuple[float, float, float]
                          ) -> Tuple[float, float]:
        """
        Compute velocity command to move toward the next waypoint.

        Simple proportional controller: turn toward waypoint, then drive.
        """
        waypoint = self.goal_selector.get_next_waypoint(
            (pose[0], pose[1]), lookahead=0.2
        )

        if waypoint is None:
            # No goal — drive forward aggressively with alternating sweeps
            self._wander_timer = getattr(self, '_wander_timer', 0) + 1
            if self._wander_timer % 40 < 20:
                return 1.2, 0.8   # Fast arc right
            else:
                return 1.2, -0.8  # Fast arc left

        # Angle to waypoint
        dx = waypoint[0] - pose[0]
        dy = waypoint[1] - pose[1]
        dist = np.sqrt(dx**2 + dy**2)
        target_angle = np.arctan2(dy, dx)

        # Angle error
        angle_error = target_angle - pose[2]
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi

        # Proportional control
        max_v = self.config.robot.max_linear_velocity
        max_omega = self.config.robot.max_angular_velocity

        if abs(angle_error) > 0.8:
            # Sharp turn — slow down, turn fast
            v = 0.05
            omega = np.clip(angle_error * 2.5, -max_omega, max_omega)
        elif abs(angle_error) > 0.3:
            # Moderate turn — drive and steer
            v = max_v * 0.5
            omega = np.clip(angle_error * 2.0, -max_omega, max_omega)
        else:
            # Aligned — full speed
            v = max_v * min(dist / 2.0, 1.0)  # Slow near goal
            v = max(v, 0.15)
            omega = np.clip(angle_error * 1.5, -max_omega, max_omega)

        return v, omega

    def get_history_dataframe(self) -> pd.DataFrame:
        """Return per-timestep metrics as a pandas DataFrame."""
        return pd.DataFrame(self.metrics.get_history_as_dict())


class ExperimentExecutor:
    """
    Executes a batch of experiment runs and saves results.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(os.path.join(results_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "summary"), exist_ok=True)

    def execute_runs(self, runs: list,
                     experiment_id: int,
                     progress: bool = True) -> pd.DataFrame:
        """
        Execute a list of ExperimentRuns and save results.

        Args:
            runs: List of ExperimentRun objects.
            experiment_id: Experiment identifier.
            progress: Show progress bars.

        Returns:
            DataFrame with summary metrics for all runs.
        """
        all_results = []

        outer_iter = runs
        if progress:
            outer_iter = tqdm(runs, desc=f"Experiment {experiment_id}")

        for run in outer_iter:
            if progress and hasattr(outer_iter, 'set_postfix'):
                outer_iter.set_postfix(name=run.name[:30])

            try:
                result = self._execute_single(run, progress=False)
                result.update(run.tags)
                all_results.append(result)

                # Save per-run CSV
                self._save_run_result(run, result)

            except Exception as e:
                print(f"\n❌ Run '{run.name}' failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Compile summary
        if all_results:
            summary_df = pd.DataFrame(all_results)
            summary_path = os.path.join(
                self.results_dir, "summary",
                f"experiment_{experiment_id}_summary.csv"
            )
            summary_df.to_csv(summary_path, index=False)
            print(f"\n📊 Summary saved: {summary_path}")
            return summary_df

        return pd.DataFrame()

    def _execute_single(self, run: ExperimentRun,
                        progress: bool = True) -> Dict:
        """Execute a single experiment run."""
        runner = SimulationRunner(run.config, seed=run.seed)
        result = runner.run(progress=progress)

        # Save timestep history
        history_df = runner.get_history_dataframe()
        if not history_df.empty:
            history_path = os.path.join(
                self.results_dir, "raw",
                f"{run.name}_history.csv"
            )
            history_df.to_csv(history_path, index=False)

        return result

    def _save_run_result(self, run: ExperimentRun, result: Dict):
        """Save a single run's final metrics."""
        result_path = os.path.join(
            self.results_dir, "raw",
            f"{run.name}_final.json"
        )
        # Convert numpy types for JSON serialization
        clean = {}
        for k, v in result.items():
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            else:
                clean[k] = v

        with open(result_path, 'w') as f:
            json.dump(clean, f, indent=2, default=str)
