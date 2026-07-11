"""
GridWorld — 2D environment representation for robot navigation simulation.

Loads environment layouts from JSON map definitions and provides:
- Binary occupancy grid (True=occupied, False=free)
- Collision checking for robot
- Dynamic obstacle support (add/remove at specified timesteps)
- Visualization via matplotlib
"""

import json
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass


# Path to the maps/ directory relative to this file
_MAPS_DIR = os.path.join(os.path.dirname(__file__), "maps")


@dataclass
class DynamicEvent:
    """A scheduled change to the environment."""
    timestep: int
    action: str               # "add" or "remove"
    obstacle: Optional[Dict]  # Obstacle spec for "add" actions
    obstacle_index: Optional[int]  # Index for "remove" actions
    description: str = ""


class GridWorld:
    """
    2D binary occupancy grid environment.

    The grid is a 2D numpy array where:
        - 1 (True) = occupied (wall/obstacle)
        - 0 (False) = free space

    Coordinates:
        - World coordinates: (x, y) in meters, origin at bottom-left
        - Grid coordinates: (row, col), origin at top-left
        - Conversion: row = height_cells - 1 - int(y / resolution)
                       col = int(x / resolution)
    """

    def __init__(self, map_name: str = "simple_room",
                 resolution: Optional[float] = None):
        """
        Load an environment from a JSON map file.

        Args:
            map_name: Name of the map (without .json extension).
            resolution: Override the map's default resolution (m/cell).
        """
        map_path = os.path.join(_MAPS_DIR, f"{map_name}.json")
        if not os.path.exists(map_path):
            raise FileNotFoundError(
                f"Map '{map_name}' not found at {map_path}. "
                f"Available maps: {self.list_available_maps()}"
            )

        with open(map_path, 'r') as f:
            self._map_data = json.load(f)

        self.name = self._map_data["name"]
        self.description = self._map_data.get("description", "")
        self.width_m = self._map_data["width_m"]
        self.height_m = self._map_data["height_m"]
        self.resolution = resolution or self._map_data.get("resolution", 0.1)

        # Grid dimensions
        self.width_cells = int(self.width_m / self.resolution)
        self.height_cells = int(self.height_m / self.resolution)

        # Robot start position
        start = self._map_data.get("robot_start", {})
        self.robot_start_x = start.get("x", self.width_m / 2)
        self.robot_start_y = start.get("y", self.height_m / 2)
        self.robot_start_theta = start.get("theta", 0.0)

        # Current list of obstacles (mutable for dynamic environments)
        self._static_walls = copy.deepcopy(self._map_data.get("walls", []))
        self._obstacles = copy.deepcopy(self._map_data.get("obstacles", []))
        self._original_obstacles = copy.deepcopy(self._obstacles)

        # Dynamic events
        self._dynamic_events = []
        for evt in self._map_data.get("dynamic_events", []):
            self._dynamic_events.append(DynamicEvent(
                timestep=evt["timestep"],
                action=evt["action"],
                obstacle=evt.get("obstacle"),
                obstacle_index=evt.get("obstacle_index"),
                description=evt.get("description", ""),
            ))
        self._dynamic_events.sort(key=lambda e: e.timestep)
        self._applied_events = set()

        # Build the grid
        self.grid = self._build_grid()

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _build_grid(self) -> np.ndarray:
        """Construct the binary occupancy grid from walls + obstacles."""
        grid = np.zeros((self.height_cells, self.width_cells), dtype=np.float32)

        # Draw walls
        for wall in self._static_walls:
            self._draw_shape(grid, wall)

        # Draw obstacles
        for obs in self._obstacles:
            self._draw_shape(grid, obs)

        return grid

    def _draw_shape(self, grid: np.ndarray, shape: Dict[str, Any]):
        """Draw a shape (rect or circle) onto the grid."""
        shape_type = shape.get("type", "rect")

        if shape_type == "rect":
            x, y = shape["x"], shape["y"]
            w, h = shape["w"], shape["h"]
            r_min, c_min = self.world_to_grid(x, y + h)
            r_max, c_max = self.world_to_grid(x + w, y)
            r_min = max(0, min(r_min, self.height_cells - 1))
            r_max = max(0, min(r_max, self.height_cells - 1))
            c_min = max(0, min(c_min, self.width_cells - 1))
            c_max = max(0, min(c_max, self.width_cells - 1))
            grid[r_min:r_max + 1, c_min:c_max + 1] = 1.0

        elif shape_type == "circle":
            cx, cy = shape["cx"], shape["cy"]
            r = shape["r"]
            # Iterate over bounding box in grid coords
            r_center, c_center = self.world_to_grid(cx, cy)
            radius_cells = int(r / self.resolution) + 1
            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    rr = r_center + dr
                    cc = c_center + dc
                    if 0 <= rr < self.height_cells and 0 <= cc < self.width_cells:
                        # Check if cell center is within circle
                        wx, wy = self.grid_to_world(rr, cc)
                        dist = np.sqrt((wx - cx) ** 2 + (wy - cy) ** 2)
                        if dist <= r:
                            grid[rr, cc] = 1.0

    def _erase_shape(self, grid: np.ndarray, shape: Dict[str, Any]):
        """Erase a shape from the grid (set to free)."""
        shape_type = shape.get("type", "rect")

        if shape_type == "rect":
            x, y = shape["x"], shape["y"]
            w, h = shape["w"], shape["h"]
            r_min, c_min = self.world_to_grid(x, y + h)
            r_max, c_max = self.world_to_grid(x + w, y)
            r_min = max(0, min(r_min, self.height_cells - 1))
            r_max = max(0, min(r_max, self.height_cells - 1))
            c_min = max(0, min(c_min, self.width_cells - 1))
            c_max = max(0, min(c_max, self.width_cells - 1))
            grid[r_min:r_max + 1, c_min:c_max + 1] = 0.0

        elif shape_type == "circle":
            cx, cy = shape["cx"], shape["cy"]
            r = shape["r"]
            r_center, c_center = self.world_to_grid(cx, cy)
            radius_cells = int(r / self.resolution) + 1
            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    rr = r_center + dr
                    cc = c_center + dc
                    if 0 <= rr < self.height_cells and 0 <= cc < self.width_cells:
                        wx, wy = self.grid_to_world(rr, cc)
                        dist = np.sqrt((wx - cx) ** 2 + (wy - cy) ** 2)
                        if dist <= r:
                            grid[rr, cc] = 0.0

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates (meters) to grid indices (row, col).

        Args:
            x: X position in meters (horizontal, left→right).
            y: Y position in meters (vertical, bottom→top).

        Returns:
            (row, col) grid indices.
        """
        col = int(x / self.resolution)
        row = self.height_cells - 1 - int(y / self.resolution)
        return (row, col)

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """
        Convert grid indices to world coordinates (center of cell).

        Args:
            row: Row index.
            col: Column index.

        Returns:
            (x, y) world coordinates in meters.
        """
        x = (col + 0.5) * self.resolution
        y = (self.height_cells - 1 - row + 0.5) * self.resolution
        return (x, y)

    # ------------------------------------------------------------------
    # Collision / occupancy queries
    # ------------------------------------------------------------------

    def is_occupied(self, x: float, y: float) -> bool:
        """Check if a world coordinate is occupied."""
        row, col = self.world_to_grid(x, y)
        if not self.is_in_bounds_grid(row, col):
            return True  # Out of bounds treated as occupied
        return self.grid[row, col] > 0.5

    def is_in_bounds(self, x: float, y: float) -> bool:
        """Check if world coordinates are within the environment."""
        return 0 <= x < self.width_m and 0 <= y < self.height_m

    def is_in_bounds_grid(self, row: int, col: int) -> bool:
        """Check if grid indices are within bounds."""
        return 0 <= row < self.height_cells and 0 <= col < self.width_cells

    def check_collision(self, x: float, y: float, radius: float = 0.15) -> bool:
        """
        Check if a circular robot at (x, y) with given radius collides.

        Args:
            x, y: Robot center in world coordinates.
            radius: Robot radius in meters.

        Returns:
            True if collision detected.
        """
        # Check points around the robot perimeter
        steps = 8
        for i in range(steps):
            angle = 2.0 * np.pi * i / steps
            px = x + radius * np.cos(angle)
            py = y + radius * np.sin(angle)
            if self.is_occupied(px, py):
                return True
        # Also check center
        return self.is_occupied(x, y)

    def get_free_cells_count(self) -> int:
        """Return the number of free (non-occupied) cells."""
        return int(np.sum(self.grid < 0.5))

    def get_occupied_cells_count(self) -> int:
        """Return the number of occupied cells."""
        return int(np.sum(self.grid >= 0.5))

    # ------------------------------------------------------------------
    # Dynamic environment
    # ------------------------------------------------------------------

    def update(self, timestep: int):
        """
        Apply any dynamic events scheduled for this timestep.

        Args:
            timestep: Current simulation timestep.

        Returns:
            List of descriptions of events applied.
        """
        applied = []
        events_applied = False
        for i, event in enumerate(self._dynamic_events):
            if i in self._applied_events:
                continue
            if event.timestep == timestep:
                self._apply_event(event)
                self._applied_events.add(i)
                applied.append(event.description)
                events_applied = True
        
        if events_applied:
            self.grid = self._build_grid()
            
        return applied

    def _apply_event(self, event: DynamicEvent):
        """Apply a single dynamic event to the grid's internal lists."""
        if event.action == "add" and event.obstacle is not None:
            self._obstacles.append(event.obstacle)

        elif event.action == "remove" and event.obstacle_index is not None:
            idx = event.obstacle_index
            if 0 <= idx < len(self._obstacles):
                self._obstacles.pop(idx)

    def spawn_obstacle(self, x: float, y: float, w: float, h: float):
        """API-driven dynamic obstacle spawning."""
        obs = {"type": "rect", "x": x, "y": y, "w": w, "h": h}
        self._obstacles.append(obs)
        self._draw_shape(self.grid, obs)

    def clear_dynamic_obstacles(self):
        """Remove all obstacles that weren't part of the original map."""
        self._obstacles = copy.deepcopy(self._original_obstacles)
        self._rebuild_grid_safe()

    def _rebuild_grid_safe(self):
        """Rebuild the entire grid from current walls + obstacles."""
        self.grid = self._build_grid()

    def reset(self):
        """Reset environment to initial state."""
        self._obstacles = copy.deepcopy(self._original_obstacles)
        self._applied_events.clear()
        self.grid = self._build_grid()

    def has_dynamic_events(self) -> bool:
        """Check if this environment has any dynamic events."""
        return len(self._dynamic_events) > 0

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def render(self, ax: Optional[plt.Axes] = None,
               robot_pose: Optional[Tuple[float, float, float]] = None,
               title: Optional[str] = None,
               show: bool = True) -> Optional[plt.Axes]:
        """
        Render the environment grid using matplotlib.

        Args:
            ax: Matplotlib axes to draw on (creates new if None).
            robot_pose: Optional (x, y, theta) to draw robot.
            title: Plot title.
            show: Whether to call plt.show().

        Returns:
            The matplotlib Axes object.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Custom colormap: white=free, dark gray=occupied
        cmap = ListedColormap(['#F0F0F0', '#2D2D2D'])
        ax.imshow(self.grid, cmap=cmap, origin='upper',
                  extent=[0, self.width_m, 0, self.height_m],
                  vmin=0, vmax=1, aspect='equal')

        # Draw robot if pose given
        if robot_pose is not None:
            x, y, theta = robot_pose
            ax.plot(x, y, 'ro', markersize=8, zorder=5)
            # Draw heading arrow
            dx = 0.5 * np.cos(theta)
            dy = 0.5 * np.sin(theta)
            ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.1,
                     fc='red', ec='red', zorder=5)

        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title(title or f"Environment: {self.name}")
        ax.set_xlim(0, self.width_m)
        ax.set_ylim(0, self.height_m)
        ax.grid(True, alpha=0.2)

        if show:
            plt.tight_layout()
            plt.show()

        return ax

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def list_available_maps() -> List[str]:
        """List all available map names."""
        maps = []
        if os.path.exists(_MAPS_DIR):
            for f in os.listdir(_MAPS_DIR):
                if f.endswith(".json"):
                    maps.append(f.replace(".json", ""))
        return sorted(maps)

    def get_ground_truth_grid(self) -> np.ndarray:
        """Return a copy of the ground truth binary grid."""
        return self.grid.copy()

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape (rows, cols)."""
        return (self.height_cells, self.width_cells)

    def __repr__(self) -> str:
        return (
            f"GridWorld(name='{self.name}', "
            f"size={self.width_m}x{self.height_m}m, "
            f"grid={self.height_cells}x{self.width_cells}, "
            f"res={self.resolution}m/cell, "
            f"obstacles={len(self._obstacles)}, "
            f"dynamic_events={len(self._dynamic_events)})"
        )
