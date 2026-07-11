import asyncio
import threading
import time
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config.settings import SimulationConfig, DecayModelType
from experiments.runner import SimulationRunner

app = FastAPI()

# Serve static files for the frontend
app.mount("/static", StaticFiles(directory="web"), name="static")


class BouncingObstacle:
    def __init__(self, x, y, w, h, vx, vy):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.vx = vx
        self.vy = vy

class SimState:
    """Manages the simulation in a background thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.t = 0
        self.is_running = False
        
        self.bouncers = [
            BouncingObstacle(15, 10, 1.5, 1.5, 0.4, 0.2),
            BouncingObstacle(30, 15, 1.5, 1.5, -0.3, 0.5),
            BouncingObstacle(5, 25, 1.5, 1.5, 0.4, -0.3)
        ]

        # Main active runner for the large map
        cfg = SimulationConfig.for_environment("office_maze")
        cfg.lidar.max_range = 15.0
        cfg.decay.model_type = DecayModelType.EXPONENTIAL
        cfg.decay.decay_lambda = 0.05
        cfg.experiment.max_timesteps = 9999999
        cfg.particle_filter.num_particles = 5
        cfg.lidar.num_beams = 90
        cfg.grid.resolution = 0.25
        self.runner = SimulationRunner(cfg, seed=42)

        # 4 background runners for comparison mini-maps
        self.comp_runners = {}
        models = [
            ("exponential", DecayModelType.EXPONENTIAL),
            ("none", DecayModelType.NONE),
            ("adaptive", DecayModelType.ADAPTIVE),
            ("aggressive", DecayModelType.AGGRESSIVE),
        ]
        for name, mtype in models:
            c = SimulationConfig.for_environment("office_maze")
            c.lidar.max_range = 15.0
            c.decay.model_type = mtype
            c.decay.decay_lambda = 0.05
            if mtype == DecayModelType.AGGRESSIVE:
                c.decay.aggressive_lambda = 0.2
            c.experiment.max_timesteps = 9999999
            c.particle_filter.num_particles = 1
            c.lidar.num_beams = 90
            c.grid.resolution = 0.25
            self.comp_runners[name] = SimulationRunner(c, seed=42)
        self.thread = threading.Thread(target=self._sim_loop, daemon=True)

    def start(self):
        self.is_running = True
        self.thread.start()

    def _sim_loop(self):
        while True:
            if self.is_running:
                with self.lock:
                    # Update dynamic moving obstacles
                    gw = self.runner.grid_world
                    for b in self.bouncers:
                        gw._erase_shape(gw.grid, {"type": "rect", "x": b.x, "y": b.y, "w": b.w, "h": b.h})
                    
                    for b in self.bouncers:
                        b.x += b.vx
                        b.y += b.vy
                        if b.x <= 0 or b.x + b.w >= gw.width_m: b.vx *= -1
                        if b.y <= 0 or b.y + b.h >= gw.height_m: b.vy *= -1
                        gw._draw_shape(gw.grid, {"type": "rect", "x": b.x, "y": b.y, "w": b.w, "h": b.h})
                        
                    # Sync ground truth across runners
                    new_gt = gw.grid.copy()
                    for r in self.comp_runners.values():
                        r.grid_world.grid = new_gt.copy()

                    # Update Main Runner
                    self.runner._timestep = self.t
                    self.runner._step(self.t)
                    
                    # Massively Optimized Comparison Runners Update
                    # They don't need to run A* path planning or raycasting!
                    # We just copy the main robot's pose and scan and apply memory decay.
                    main_pose = self.runner.particle_filter.get_best_pose()
                    main_scan = self.runner.lidar.scan(self.runner.robot.true_pose, self.runner.grid_world)
                    
                    for r in self.comp_runners.values():
                        # Update their local ground truth so obstacle bounds match
                        r.grid_world.grid = new_gt.copy()
                        r_grid = r.particle_filter.get_occupancy_grid()
                        # Apply identical sensor reading
                        r_grid.update(main_pose, main_scan, self.t)
                        # Apply custom decay
                        if r.memory_manager.should_apply_decay(self.t):
                            r.memory_manager.apply_decay(r_grid, float(self.t))
                        r._timestep = self.t
                        
                    self.t += 1
            # Adjust sleep time based on simulation speed (default 20 steps/sec)
            sleep_time = 0.05 / getattr(self, 'sim_speed', 1.0)
            time.sleep(sleep_time)


sim_state = SimState()


@app.on_event("startup")
def startup_event():
    sim_state.start()


@app.get("/")
def read_root():
    return FileResponse("web/index.html")


# ── REST Endpoints ──────────────────────────────────────────────

class ObstacleReq(BaseModel):
    x: float
    y: float
    w: float
    h: float


@app.post("/api/spawn_obstacle")
def spawn_obstacle(req: ObstacleReq):
    with sim_state.lock:
        sim_state.runner.grid_world.spawn_obstacle(req.x, req.y, req.w, req.h)
        for r in sim_state.comp_runners.values():
            r.grid_world.spawn_obstacle(req.x, req.y, req.w, req.h)
    return {"status": "ok"}


@app.post("/api/clear_memory")
def clear_memory():
    with sim_state.lock:
        sim_state.runner.particle_filter.get_occupancy_grid().reset()
        for r in sim_state.comp_runners.values():
            r.particle_filter.get_occupancy_grid().reset()
            
        # Pause simulation briefly so the user sees the map wipe
        sim_state.is_running = False

    # Resume after 1 second (let a few blank frames render)
    def _resume():
        import time as _t
        _t.sleep(1.0)
        sim_state.is_running = True

    threading.Thread(target=_resume, daemon=True).start()
    return {"status": "ok"}


class DecayReq(BaseModel):
    rate: float
    model: str = None


@app.post("/api/set_decay")
def set_decay(req: DecayReq):
    with sim_state.lock:
        if req.model:
            # Change the model completely
            from memory.decay_models import create_decay_model
            model_type = DecayModelType(req.model)
            sim_state.runner.config.decay.model_type = model_type
            sim_state.runner.config.decay.decay_lambda = req.rate
            sim_state.runner.memory_manager.config.model_type = model_type
            sim_state.runner.memory_manager.config.decay_lambda = req.rate
            sim_state.runner.memory_manager.decay_model = create_decay_model(
                req.model, decay_lambda=req.rate
            )
        else:
            # Just update rate
            sim_state.runner.config.decay.decay_lambda = req.rate
            sim_state.runner.memory_manager.config.decay_lambda = req.rate
            model = sim_state.runner.memory_manager.decay_model
            if hasattr(model, 'decay_lambda'):
                model.decay_lambda = req.rate
    return {"status": "ok"}


class SpeedReq(BaseModel):
    speed: float

@app.post("/api/set_speed")
def set_speed(req: SpeedReq):
    sim_state.sim_speed = req.speed
    return {"status": "ok"}


@app.post("/api/clear_obstacles")
def clear_obstacles():
    with sim_state.lock:
        def _clear_obs(runner):
            gw = runner.grid_world
            grid_obj = runner.particle_filter.get_occupancy_grid()
            old_gt = gw.grid.copy()
            gw.clear_dynamic_obstacles()
            changed = (old_gt == 1) & (gw.grid == 0)
            grid_obj.log_odds[changed] = grid_obj.grid_config.log_odds_prior
            grid_obj.last_observed[changed] = -1.0
            grid_obj.visit_count[changed] = 0
            
        _clear_obs(sim_state.runner)
        for r in sim_state.comp_runners.values():
            _clear_obs(r)

    return {"status": "ok"}


@app.post("/api/toggle")
def toggle_sim():
    sim_state.is_running = not sim_state.is_running
    return {"status": "ok", "running": sim_state.is_running}


# ── WebSocket Stream ────────────────────────────────────────────

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with sim_state.lock:
                grid_obj = sim_state.runner.particle_filter.get_occupancy_grid()
                prob_map = grid_obj.get_probability_map()
                last_obs = grid_obj.last_observed

                # Build a combined array:
                #   -1 = never observed (unknown)
                #   0..1 = probability (0=free, 1=occupied)
                combined = np.where(last_obs >= 0, prob_map, -1.0)
                flat = [round(float(v), 2) for v in combined.flatten()]

                # Ground truth for overlay
                gt = sim_state.runner.grid_world.grid
                gt_flat = [int(v) for v in gt.flatten()]

                # Frontiers
                try:
                    reex = sim_state.runner.memory_manager.get_reexploration_candidates(grid_obj)
                    frontiers = sim_state.runner.frontier_explorer.detect_frontiers(grid_obj, reex)
                    frontier_pts = [list(f.centroid_grid) for f in frontiers]
                except Exception:
                    frontier_pts = []

                # Path
                try:
                    path = sim_state.runner.goal_selector.current_path or []
                except Exception:
                    path = []

                # We also want to send the last_observed timestamps to render fading colors
                last_obs_list = [int(v) for v in last_obs.flatten()]

                data = {
                    "t": sim_state.t,
                    "grid": flat,
                    "gt": gt_flat,
                    "last_obs": last_obs_list,
                    "w": int(prob_map.shape[1]),
                    "h": int(prob_map.shape[0]),
                    "res": sim_state.runner.grid_world.resolution,
                    "est": list(sim_state.runner.particle_filter.get_best_pose()),
                    "true": list(sim_state.runner.robot.true_pose),
                    "path": path,
                    "frontiers": frontier_pts,
                    "comps": {}
                }
                
                # Send comparison grids every 5 frames to save WS bandwidth
                if sim_state.t % 5 == 0:
                    for name, runner in sim_state.comp_runners.items():
                        c_grid_obj = runner.particle_filter.get_occupancy_grid()
                        c_prob = c_grid_obj.get_probability_map()
                        c_last = c_grid_obj.last_observed
                        c_combined = np.where(c_last >= 0, c_prob, -1.0)
                        
                        # Downsample (stride of 2) to further save bandwidth for the small minimaps
                        c_down = c_combined[::2, ::2]
                        c_last_down = c_last[::2, ::2]
                        
                        data["comps"][name] = {
                            "grid": [round(float(v), 2) for v in c_down.flatten()],
                            "last_obs": [int(v) for v in c_last_down.flatten()],
                            "w": int(c_down.shape[1]),
                            "h": int(c_down.shape[0])
                        }

            await websocket.send_json(data)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
