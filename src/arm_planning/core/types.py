from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Pose:
    position: Array
    quat_wxyz: Array | None = None


@dataclass(frozen=True)
class ObstacleSpec:
    type: str
    name: str
    position: Array
    size: Array | None = None
    radius: float | None = None
    height: float | None = None
    quat_wxyz: Array | None = None


@dataclass(frozen=True)
class SceneSpec:
    id: str
    seed: int
    q_start: Array
    target_position: Array
    obstacles: list[ObstacleSpec]
    target_quat_wxyz: Array | None = None


@dataclass(frozen=True)
class RobotSpec:
    name: str
    model_xml: str
    ee_site: str
    joint_names: list[str]
    q_start: Array


@dataclass(frozen=True)
class ProjectConfig:
    global_seed: int
    robot: RobotSpec
    scenes: list[SceneSpec]
    experiment: dict[str, Any]
    ik: dict[str, Any]
    planners: dict[str, Any]
    trajectory: dict[str, Any]
    rerun: dict[str, Any]


@dataclass
class IKResult:
    success: bool
    q: Array
    position_error: float
    orientation_error: float
    iterations: int
    solve_time: float
    method: str
    condition_number: float
    message: str = ""


@dataclass
class PlanResult:
    success: bool
    path: list[Array]
    planning_time: float
    path_length: float
    collision_checks: int
    planner_name: str
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialMetrics:
    success: bool
    planning_time: float
    path_length_joint: float
    path_length_task: float
    smoothness: float
    min_obstacle_distance: float
    collision_checks: int
    num_waypoints: int
    final_error: float
    metadata: dict[str, Any] = field(default_factory=dict)
