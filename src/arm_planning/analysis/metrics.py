from __future__ import annotations

import numpy as np

from arm_planning.core.math_utils import joint_path_length
from arm_planning.core.types import TrialMetrics
from arm_planning.ik.solvers import orientation_error_vector
from arm_planning.sim.mujoco_world import MujocoWorld


def task_path_length(ee_points: np.ndarray) -> float:
    if len(ee_points) < 2:
        return 0.0
    return float(sum(np.linalg.norm(ee_points[i + 1] - ee_points[i]) for i in range(len(ee_points) - 1)))


def smoothness(path: list[np.ndarray]) -> float:
    if len(path) < 3:
        return 0.0
    return float(sum(np.linalg.norm(path[i + 1] - 2.0 * path[i] + path[i - 1]) for i in range(1, len(path) - 1)))


def min_obstacle_distance(path: list[np.ndarray], context: MujocoWorld) -> float:
    if not path:
        return float("nan")
    return float(min(context.min_obstacle_distance(q) for q in path))


def min_ground_clearance(path: list[np.ndarray], context: MujocoWorld) -> float:
    if not path:
        return float("nan")
    if not hasattr(context, "ground_clearance"):
        return float("inf")
    return float(min(context.ground_clearance(q) for q in path))


def trajectory_kinematic_metrics(path: list[np.ndarray], context: MujocoWorld, target_position: np.ndarray, target_quat_wxyz: np.ndarray | None = None) -> dict[str, float]:
    if not path:
        return {
            "max_joint_step": float("nan"),
            "mean_joint_step": float("nan"),
            "max_joint_speed_norm": float("nan"),
            "mean_joint_speed_norm": float("nan"),
            "max_joint_acc_norm": float("nan"),
            "mean_joint_acc_norm": float("nan"),
            "max_joint_jerk_norm": float("nan"),
            "mean_joint_jerk_norm": float("nan"),
            "path_length_task": float("nan"),
            "smoothness": float("nan"),
            "min_obstacle_distance": float("nan"),
            "min_ground_clearance": float("nan"),
            "safety_margin": float("nan"),
            "final_error": float("nan"),
            "final_orientation_error": float("nan"),
        }
    arr = np.asarray(path, dtype=float)
    if len(arr) >= 2:
        dq = np.diff(arr, axis=0)
        speed_norm = np.linalg.norm(dq, axis=1)
        max_joint_step = float(np.max(speed_norm))
        mean_joint_step = float(np.mean(speed_norm))
    else:
        dq = np.empty((0, arr.shape[1]))
        speed_norm = np.asarray([0.0])
        max_joint_step = 0.0
        mean_joint_step = 0.0
    if len(arr) >= 3:
        ddq = np.diff(arr, n=2, axis=0)
        acc_norm = np.linalg.norm(ddq, axis=1)
    else:
        acc_norm = np.asarray([0.0])
    if len(arr) >= 4:
        dddq = np.diff(arr, n=3, axis=0)
        jerk_norm = np.linalg.norm(dddq, axis=1)
    else:
        jerk_norm = np.asarray([0.0])
    ee_points = context.sample_end_effector_path(path)
    min_distance = min_obstacle_distance(path, context)
    ground_clearance = min_ground_clearance(path, context)
    if target_quat_wxyz is None:
        final_orientation_error = 0.0
    else:
        final_orientation_error = float(np.linalg.norm(orientation_error_vector(context.forward_quat(path[-1]), target_quat_wxyz)))
    return {
        "max_joint_step": max_joint_step,
        "mean_joint_step": mean_joint_step,
        "max_joint_speed_norm": float(np.max(speed_norm)),
        "mean_joint_speed_norm": float(np.mean(speed_norm)),
        "max_joint_acc_norm": float(np.max(acc_norm)),
        "mean_joint_acc_norm": float(np.mean(acc_norm)),
        "max_joint_jerk_norm": float(np.max(jerk_norm)),
        "mean_joint_jerk_norm": float(np.mean(jerk_norm)),
        "path_length_task": task_path_length(ee_points),
        "smoothness": smoothness(path),
        "min_obstacle_distance": min_distance,
        "min_ground_clearance": ground_clearance,
        "safety_margin": min_distance,
        "final_error": float(np.linalg.norm(ee_points[-1] - target_position)),
        "final_orientation_error": final_orientation_error,
    }


def compute_trial_metrics(path: list[np.ndarray], context: MujocoWorld, target_position: np.ndarray, planning_time: float, collision_checks: int, success: bool, target_quat_wxyz: np.ndarray | None = None) -> TrialMetrics:
    if path:
        ee_points = context.sample_end_effector_path(path)
        final_error = float(np.linalg.norm(ee_points[-1] - target_position))
        task_len = task_path_length(ee_points)
        min_dist = min_obstacle_distance(path, context)
        ground_clearance = min_ground_clearance(path, context)
    else:
        final_error = float("nan")
        task_len = float("nan")
        min_dist = float("nan")
        ground_clearance = float("nan")
    return TrialMetrics(
        success=bool(success),
        planning_time=float(planning_time),
        path_length_joint=joint_path_length(path),
        path_length_task=task_len,
        smoothness=smoothness(path),
        min_obstacle_distance=min_dist,
        collision_checks=int(collision_checks),
        num_waypoints=len(path),
        final_error=final_error,
        metadata={"min_ground_clearance": ground_clearance},
    )
