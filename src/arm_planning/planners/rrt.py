from __future__ import annotations

import time

import numpy as np

from arm_planning.core.math_utils import joint_path_length, nearest_index
from arm_planning.core.types import PlanResult
from arm_planning.planners.base import Planner
from arm_planning.sim.mujoco_world import MujocoWorld


def _sample(context: MujocoWorld, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(context.lower_limits, context.upper_limits)


def _steer(q_from: np.ndarray, q_to: np.ndarray, step_size: float) -> np.ndarray:
    delta = q_to - q_from
    dist = float(np.linalg.norm(delta))
    if dist <= step_size:
        return q_to.copy()
    return q_from + delta / dist * step_size


def _backtrack(nodes: list[np.ndarray], parents: list[int], idx: int) -> list[np.ndarray]:
    path = []
    while idx != -1:
        path.append(nodes[idx])
        idx = parents[idx]
    return list(reversed(path))


class RRTPlanner(Planner):
    name = "rrt"

    def __init__(self, params: dict):
        self.max_iterations = int(params.get("max_iterations", 700))
        self.step_size = float(params.get("step_size", 0.20))
        self.goal_tolerance = float(params.get("goal_tolerance", 0.22))
        self.goal_bias = float(params.get("goal_bias", 0.18))

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray, context: MujocoWorld, rng: np.random.Generator) -> PlanResult:
        start_time = time.perf_counter()
        start_checks = context.collision_checks
        if not context.is_state_valid(q_start) or not context.is_state_valid(q_goal):
            return PlanResult(False, [], time.perf_counter() - start_time, 0.0, context.collision_checks - start_checks, self.name, "invalid start or goal")
        nodes = [q_start.copy()]
        parents = [-1]
        best_q = q_start.copy()
        best_goal_dist = float(np.linalg.norm(q_start - q_goal))
        for _ in range(self.max_iterations):
            q_rand = q_goal if rng.random() < self.goal_bias else _sample(context, rng)
            idx = nearest_index(nodes, q_rand)
            q_new = _steer(nodes[idx], q_rand, self.step_size)
            if not context.is_edge_valid(nodes[idx], q_new):
                continue
            nodes.append(q_new)
            parents.append(idx)
            goal_dist = float(np.linalg.norm(q_new - q_goal))
            if goal_dist < best_goal_dist:
                best_goal_dist = goal_dist
                best_q = q_new.copy()
            if np.linalg.norm(q_new - q_goal) <= self.goal_tolerance and context.is_edge_valid(q_new, q_goal):
                path = _backtrack(nodes, parents, len(nodes) - 1) + [q_goal.copy()]
                return PlanResult(True, path, time.perf_counter() - start_time, joint_path_length(path), context.collision_checks - start_checks, self.name, "success")
        return PlanResult(
            False,
            [],
            time.perf_counter() - start_time,
            0.0,
            context.collision_checks - start_checks,
            self.name,
            "max iterations reached",
            {"failure_q": best_q.copy()},
        )
