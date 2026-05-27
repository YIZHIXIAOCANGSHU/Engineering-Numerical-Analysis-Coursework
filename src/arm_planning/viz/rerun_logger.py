from __future__ import annotations

from pathlib import Path

import numpy as np

from arm_planning.core.types import SceneSpec, TrialMetrics
from arm_planning.utils.config import resolve_path


class RerunLogger:
    def __init__(self, enabled: bool = True, recording_path: str | Path = "results/recordings/arm_planning.rrd", spawn: bool = False):
        self.enabled = bool(enabled)
        self.recording_path = resolve_path(recording_path)
        self.spawn = bool(spawn)
        self._rr = None
        if self.enabled:
            import rerun as rr

            self._rr = rr
            self.recording_path.parent.mkdir(parents=True, exist_ok=True)
            rr.init("arm_planning_mujoco", spawn=self.spawn)
            rr.save(self.recording_path)

    def log_scene(self, scene: SceneSpec) -> None:
        if not self.enabled:
            return
        rr = self._rr
        assert rr is not None
        rr.log("scene/target", rr.Points3D([scene.target_position], radii=[0.035], colors=[[0, 220, 80]]), static=True)
        for obs in scene.obstacles:
            if obs.type == "box" and obs.size is not None:
                rr.log(f"scene/obstacles/{obs.name}", rr.Boxes3D(centers=[obs.position], sizes=[obs.size], colors=[[230, 80, 40, 160]]), static=True)
            elif obs.type == "sphere" and obs.radius is not None:
                rr.log(f"scene/obstacles/{obs.name}", rr.Ellipsoids3D(centers=[obs.position], radii=[obs.radius], colors=[[230, 80, 40, 160]]), static=True)
            elif obs.type == "cylinder" and obs.radius is not None and obs.height is not None:
                # Rerun has no cylinder primitive in the basic archetypes; use an ellipsoid marker.
                rr.log(f"scene/obstacles/{obs.name}", rr.Ellipsoids3D(centers=[obs.position], half_sizes=[[obs.radius, obs.radius, obs.height / 2.0]], colors=[[230, 80, 40, 120]]), static=True)

    def log_path(self, planner_name: str, ee_points: np.ndarray) -> None:
        if not self.enabled or len(ee_points) == 0:
            return
        rr = self._rr
        assert rr is not None
        rr.log(f"planner/{planner_name}/ee_path", rr.LineStrips3D([ee_points], colors=[[70, 130, 240]], radii=[0.008]))

    def _set_time_sequence(self, timeline: str, value: int) -> None:
        rr = self._rr
        assert rr is not None
        if hasattr(rr, "set_time"):
            rr.set_time(timeline, sequence=value)
        else:
            rr.set_time_sequence(timeline, value)

    def log_joint_series(self, planner_name: str, path: list[np.ndarray]) -> None:
        if not self.enabled:
            return
        rr = self._rr
        assert rr is not None
        for i, q in enumerate(path):
            self._set_time_sequence("waypoint", i)
            for j, value in enumerate(q):
                rr.log(f"joints/{planner_name}/q{j + 1}", rr.Scalars(float(value)))

    def log_metrics(self, planner_name: str, metrics: TrialMetrics) -> None:
        if not self.enabled:
            return
        rr = self._rr
        assert rr is not None
        self._set_time_sequence("summary", 0)
        for key, value in metrics.__dict__.items():
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                rr.log(f"metrics/{planner_name}/{key}", rr.Scalars(float(value)))

    def path(self) -> Path:
        return self.recording_path
