"""
Live dashboard — Pygame-based real-time visualization of the simulation.

4-panel layout:
    TL = Ground truth environment
    TR = Estimated occupancy grid (color-coded by decay state)
    BL = Uncertainty heatmap
    BR = Robot trajectory + frontiers + current path

Side panel with live metrics.
"""

import numpy as np
import os

# Conditionally import pygame
try:
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from typing import Tuple, Optional, List, Dict
from slam.occupancy_grid import DecayingOccupancyGrid
from environments.grid_world import GridWorld


# Color palette
COLORS = {
    "bg":           (18, 18, 24),
    "panel_bg":     (28, 28, 38),
    "panel_border": (60, 60, 80),
    "text":         (220, 220, 230),
    "text_dim":     (140, 140, 160),
    "accent":       (100, 180, 255),
    "accent2":      (255, 140, 80),
    "robot":        (255, 80, 80),
    "path":         (80, 255, 140),
    "frontier_exp": (255, 255, 80),
    "frontier_dec": (255, 140, 255),
    "free":         (30, 30, 40),
    "occupied":     (220, 220, 230),
    "unknown":      (60, 60, 80),
    "decayed":      (120, 60, 160),
    "certain_free": (40, 100, 80),
    "certain_occ":  (200, 60, 60),
}


class LiveDashboard:
    """
    Real-time Pygame dashboard for simulation monitoring.

    Layout (1280x720):
        ┌──────────┬──────────┬────────┐
        │ Ground   │ Estimated│ Side   │
        │ Truth    │ Map      │ Panel  │
        ├──────────┼──────────┤ (live  │
        │ Uncert.  │ Trajectory│metrics)│
        │ Heatmap  │ + Path   │        │
        └──────────┴──────────┴────────┘
    """

    def __init__(self, grid_world: GridWorld,
                 width: int = 1280, height: int = 720,
                 fps: int = 30):
        """
        Args:
            grid_world: Ground truth environment.
            width: Window width in pixels.
            height: Window height in pixels.
            fps: Target frame rate.
        """
        if not HAS_PYGAME:
            raise ImportError(
                "pygame is required for LiveDashboard. "
                "Install with: pip install pygame"
            )

        self.grid_world = grid_world
        self.width = width
        self.height = height
        self.fps = fps

        # Panel dimensions
        self.side_panel_w = 240
        self.panel_w = (width - self.side_panel_w) // 2
        self.panel_h = height // 2
        self.padding = 4

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MemoryPy — Live Dashboard")
        self.clock = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("Menlo", 14, bold=True)
        self.font_sm = pygame.font.SysFont("Menlo", 11)
        self.font_title = pygame.font.SysFont("Menlo", 16, bold=True)

        # Prerender ground truth
        self._gt_surface = self._render_gt_map()

        # State
        self.running = True
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, grid: DecayingOccupancyGrid,
               true_pose: Tuple[float, float, float],
               est_pose: Tuple[float, float, float],
               trajectory: List[Tuple[float, float, float]],
               path: Optional[List[Tuple[float, float]]] = None,
               frontiers: Optional[List] = None,
               metrics: Optional[Dict] = None,
               timestep: int = 0) -> bool:
        """
        Render one frame of the dashboard.

        Args:
            grid: Current occupancy grid.
            true_pose: Ground truth pose.
            est_pose: Estimated pose.
            trajectory: List of past poses.
            path: Current planned path.
            frontiers: Detected frontiers.
            metrics: Dict of live metrics to display.
            timestep: Current timestep.

        Returns:
            False if user closed the window.
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                    return False

        self.screen.fill(COLORS["bg"])

        # Panel 1: Ground Truth (top-left)
        self._draw_panel(0, 0, "Ground Truth", self._gt_surface, true_pose)

        # Panel 2: Estimated Map (top-right)
        est_surface = self._render_estimated_map(grid)
        self._draw_panel(self.panel_w, 0, "Estimated Map", est_surface, est_pose)

        # Panel 3: Uncertainty Heatmap (bottom-left)
        unc_surface = self._render_uncertainty_map(grid)
        self._draw_panel(0, self.panel_h, "Uncertainty", unc_surface, est_pose)

        # Panel 4: Trajectory (bottom-right)
        traj_surface = self._render_trajectory(
            grid, trajectory, path, frontiers, est_pose
        )
        self._draw_panel(self.panel_w, self.panel_h, "Trajectory", traj_surface)

        # Side panel: Metrics
        self._draw_side_panel(metrics or {}, timestep)

        pygame.display.flip()
        self.clock.tick(self.fps)
        self._frame_count += 1

        return True

    # ------------------------------------------------------------------
    # Panel rendering
    # ------------------------------------------------------------------

    def _draw_panel(self, x: int, y: int, title: str,
                    surface: pygame.Surface,
                    pose: Optional[Tuple[float, float, float]] = None):
        """Draw a panel with border, title, and content."""
        p = self.padding
        pw = self.panel_w - 2 * p
        ph = self.panel_h - 2 * p
        title_h = 20

        # Border
        rect = pygame.Rect(x + p, y + p, pw, ph)
        pygame.draw.rect(self.screen, COLORS["panel_bg"], rect)
        pygame.draw.rect(self.screen, COLORS["panel_border"], rect, 1)

        # Title
        title_surf = self.font_lg.render(title, True, COLORS["accent"])
        self.screen.blit(title_surf, (x + p + 6, y + p + 2))

        # Content area
        content_rect = pygame.Rect(x + p + 2, y + p + title_h,
                                   pw - 4, ph - title_h - 2)

        # Scale surface to fit
        scaled = pygame.transform.scale(surface, (content_rect.width,
                                                   content_rect.height))
        self.screen.blit(scaled, content_rect.topleft)

        # Draw robot pose on the panel
        if pose is not None:
            rx = content_rect.x + int(pose[0] / self.grid_world.width_m * content_rect.width)
            ry = content_rect.y + content_rect.height - int(
                pose[1] / self.grid_world.height_m * content_rect.height)
            rx = max(content_rect.x, min(rx, content_rect.right - 1))
            ry = max(content_rect.y, min(ry, content_rect.bottom - 1))
            pygame.draw.circle(self.screen, COLORS["robot"], (rx, ry), 4)

            # Heading line
            dx = int(8 * np.cos(pose[2]))
            dy = int(-8 * np.sin(pose[2]))
            pygame.draw.line(self.screen, COLORS["robot"],
                             (rx, ry), (rx + dx, ry + dy), 2)

    def _draw_side_panel(self, metrics: Dict, timestep: int):
        """Draw the side metrics panel."""
        x = self.width - self.side_panel_w
        p = self.padding

        # Background
        rect = pygame.Rect(x + p, p, self.side_panel_w - 2 * p,
                           self.height - 2 * p)
        pygame.draw.rect(self.screen, COLORS["panel_bg"], rect)
        pygame.draw.rect(self.screen, COLORS["panel_border"], rect, 1)

        # Title
        title = self.font_title.render("METRICS", True, COLORS["accent"])
        self.screen.blit(title, (x + 12, 10))

        # Timestep
        y_pos = 35
        ts_text = self.font_lg.render(f"Step: {timestep}", True, COLORS["text"])
        self.screen.blit(ts_text, (x + 12, y_pos))
        y_pos += 22

        fps_text = self.font_sm.render(
            f"FPS: {self.clock.get_fps():.0f}", True, COLORS["text_dim"]
        )
        self.screen.blit(fps_text, (x + 12, y_pos))
        y_pos += 25

        # Separator
        pygame.draw.line(self.screen, COLORS["panel_border"],
                         (x + 12, y_pos), (x + self.side_panel_w - 16, y_pos))
        y_pos += 10

        # Metrics list
        metric_labels = [
            ("Coverage", "coverage", "%"),
            ("Map MSE", "map_mse", ""),
            ("Map SSIM", "map_ssim", ""),
            ("Loc. RMSE", "localization_rmse", "m"),
            ("Distance", "total_distance", "m"),
            ("Entropy", "map_entropy", ""),
            ("Re-explore", "reexploration_ratio", "%"),
            ("Nav Success", "nav_success_rate", "%"),
            ("Memory", "memory_usage", ""),
            ("Occ. Cells", "certain_occupied", ""),
            ("Free Cells", "certain_free", ""),
            ("Uncertain", "uncertain", ""),
        ]

        for label, key, unit in metric_labels:
            val = metrics.get(key)
            if val is None:
                val_str = "—"
            elif unit == "%":
                val_str = f"{val:.1%}"
            elif unit == "m":
                val_str = f"{val:.3f}{unit}"
            elif isinstance(val, float):
                val_str = f"{val:.3f}"
            else:
                val_str = f"{val}"

            # Label
            lbl = self.font_sm.render(f"{label}:", True, COLORS["text_dim"])
            self.screen.blit(lbl, (x + 12, y_pos))

            # Value
            v = self.font_sm.render(val_str, True, COLORS["text"])
            self.screen.blit(v, (x + self.side_panel_w - 16 - v.get_width(), y_pos))
            y_pos += 18

        # Decay model info
        y_pos += 10
        pygame.draw.line(self.screen, COLORS["panel_border"],
                         (x + 12, y_pos), (x + self.side_panel_w - 16, y_pos))
        y_pos += 10

        model_name = metrics.get("model_name", "")
        if model_name:
            m = self.font_sm.render(f"Decay: {model_name}", True, COLORS["accent2"])
            self.screen.blit(m, (x + 12, y_pos))
            y_pos += 18

        reex = metrics.get("reexploration_candidates", 0)
        r = self.font_sm.render(f"Re-explore: {reex} cells", True, COLORS["text_dim"])
        self.screen.blit(r, (x + 12, y_pos))

    # ------------------------------------------------------------------
    # Map rendering helpers
    # ------------------------------------------------------------------

    def _render_gt_map(self) -> pygame.Surface:
        """Render the ground truth grid as a pygame surface."""
        grid = self.grid_world.grid
        h, w = grid.shape
        surface = pygame.Surface((w, h))

        pixels = np.zeros((w, h, 3), dtype=np.uint8)
        pixels[grid.T < 0.5] = COLORS["free"]
        pixels[grid.T >= 0.5] = COLORS["occupied"]

        pygame.surfarray.blit_array(surface, pixels)
        return surface

    def _render_estimated_map(self, grid: DecayingOccupancyGrid) -> pygame.Surface:
        """Render the estimated map with decay-state coloring."""
        prob = grid.get_probability_map()
        observed = grid.last_observed >= 0
        h, w = prob.shape

        surface = pygame.Surface((w, h))
        pixels = np.zeros((w, h, 3), dtype=np.uint8)

        # Unknown (never observed)
        unknown = ~observed
        pixels[unknown.T] = COLORS["unknown"]

        # Free cells (observed, low probability)
        free = observed & (prob < 0.4)
        pixels[free.T] = COLORS["certain_free"]

        # Occupied cells
        occ = observed & (prob > 0.6)
        pixels[occ.T] = COLORS["certain_occ"]

        # Uncertain (decayed or ambiguous)
        uncertain = observed & (prob >= 0.4) & (prob <= 0.6)
        pixels[uncertain.T] = COLORS["decayed"]

        pygame.surfarray.blit_array(surface, pixels)
        return surface

    def _render_uncertainty_map(self, grid: DecayingOccupancyGrid) -> pygame.Surface:
        """Render entropy as a heatmap."""
        entropy = grid.get_uncertainty_map()
        h, w = entropy.shape

        # Normalize to 0-1
        max_e = np.log(2)  # Maximum binary entropy
        norm = np.clip(entropy / max_e, 0, 1)

        surface = pygame.Surface((w, h))
        pixels = np.zeros((w, h, 3), dtype=np.uint8)

        # Low entropy (certain) → dark blue, high entropy → bright yellow
        pixels[:, :, 0] = (norm * 255).astype(np.uint8).T  # R
        pixels[:, :, 1] = (norm * 200).astype(np.uint8).T  # G
        pixels[:, :, 2] = ((1 - norm) * 120).astype(np.uint8).T  # B

        pygame.surfarray.blit_array(surface, pixels)
        return surface

    def _render_trajectory(self, grid: DecayingOccupancyGrid,
                           trajectory: List[Tuple[float, float, float]],
                           path: Optional[List[Tuple[float, float]]],
                           frontiers: Optional[List],
                           pose: Optional[Tuple[float, float, float]]
                           ) -> pygame.Surface:
        """Render trajectory, path, and frontiers on the estimated map."""
        prob = grid.get_probability_map()
        h, w = prob.shape
        surface = pygame.Surface((w, h))

        # Background: dim version of estimated map
        pixels = np.zeros((w, h, 3), dtype=np.uint8)
        gray = (prob * 60).astype(np.uint8)
        pixels[:, :, 0] = gray.T
        pixels[:, :, 1] = gray.T
        pixels[:, :, 2] = gray.T + 20
        pygame.surfarray.blit_array(surface, pixels)

        wm = self.grid_world.width_m
        hm = self.grid_world.height_m

        def world_to_px(x, y):
            px = int(x / wm * w)
            py = h - 1 - int(y / hm * h)
            return (max(0, min(px, w-1)), max(0, min(py, h-1)))

        # Draw trajectory
        if len(trajectory) > 1:
            pts = [world_to_px(p[0], p[1]) for p in trajectory[-500:]]
            if len(pts) > 1:
                pygame.draw.lines(surface, (80, 180, 255), False, pts, 1)

        # Draw planned path
        if path and len(path) > 1:
            pts = [world_to_px(p[0], p[1]) for p in path]
            pygame.draw.lines(surface, COLORS["path"], False, pts, 2)

        # Draw frontiers
        if frontiers:
            for f in frontiers[:10]:
                cx, cy = f.centroid_world(grid)
                px, py = world_to_px(cx, cy)
                color = (COLORS["frontier_exp"] if f.frontier_type == "exploration"
                         else COLORS["frontier_dec"])
                pygame.draw.circle(surface, color, (px, py), 3)

        return surface

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def capture_frame(self) -> np.ndarray:
        """Capture the current frame as a numpy array (H, W, 3)."""
        frame = pygame.surfarray.array3d(self.screen)
        return frame.transpose(1, 0, 2)  # pygame is (W, H, 3), we want (H, W, 3)

    def close(self):
        """Close the dashboard window."""
        pygame.quit()
        self.running = False

    def __del__(self):
        if HAS_PYGAME and pygame.get_init():
            pygame.quit()
