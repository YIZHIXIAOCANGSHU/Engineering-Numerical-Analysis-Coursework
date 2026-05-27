from __future__ import annotations

import time

import numpy as np

from arm_planning.core.math_utils import joint_path_length, nearest_index
from arm_planning.core.types import PlanResult
from arm_planning.planners.base import Planner
from arm_planning.planners.rrt import _backtrack, _sample, _steer
from arm_planning.sim.mujoco_world import MujocoWorld


class RRTConnectPlanner(Planner):
    name = "rrt_connect"

    def __init__(self, params: dict):
        self.max_iterations = int(params.get("max_iterations", 700))
        self.step_size = float(params.get("step_size", 0.20))
        self.goal_tolerance = float(params.get("goal_tolerance", 0.22))

    def _extend(self, nodes: list[np.ndarray], parents: list[int], target: np.ndarray, context: MujocoWorld) -> tuple[str, int | None]:
        idx = nearest_index(nodes, target)
        q_new = _steer(nodes[idx], target, self.step_size)
        if not context.is_edge_valid(nodes[idx], q_new):
            return "trapped", None
        nodes.append(q_new)
        parents.append(idx)
        new_idx = len(nodes) - 1
        if np.linalg.norm(q_new - target) <= self.goal_tolerance:
            return "reached", new_idx
        return "advanced", new_idx

    def _connect(self, nodes: list[np.ndarray], parents: list[int], target: np.ndarray, context: MujocoWorld) -> tuple[bool, int | None]:
        last_idx = None
        for _ in range(100):
            status, idx = self._extend(nodes, parents, target, context)
            if status == "trapped":
                return False, last_idx
            last_idx = idx
            assert idx is not None
            if status == "reached":
                return True, idx
        return False, last_idx

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray, context: MujocoWorld, rng: np.random.Generator) -> PlanResult:
        start_time = time.perf_counter()
        start_checks = context.collision_checks
        if not context.is_state_valid(q_start) or not context.is_state_valid(q_goal):
            return PlanResult(False, [], time.perf_counter() - start_time, 0.0, context.collision_checks - start_checks, self.name, "invalid start or goal")
        a_nodes, a_parents = [q_start.copy()], [-1]
        b_nodes, b_parents = [q_goal.copy()], [-1]
        a_is_start = True
        best_q = q_start.copy()
        best_goal_dist = float(np.linalg.norm(q_start - q_goal))
        for _ in range(self.max_iterations):
            q_rand = _sample(context, rng)
            status, a_idx = self._extend(a_nodes, a_parents, q_rand, context)
            if status != "trapped" and a_idx is not None:
                goal_dist = float(np.linalg.norm(a_nodes[a_idx] - q_goal))
                if goal_dist < best_goal_dist:
                    best_goal_dist = goal_dist
                    best_q = a_nodes[a_idx].copy()
                reached, b_idx = self._connect(b_nodes, b_parents, a_nodes[a_idx], context)
                if reached and b_idx is not None:
                    a_path = _backtrack(a_nodes, a_parents, a_idx)
                    b_path = _backtrack(b_nodes, b_parents, b_idx)
                    if a_is_start:
                        path = a_path + list(reversed(b_path))
                    else:
                        path = b_path + list(reversed(a_path))
                    return PlanResult(True, path, time.perf_counter() - start_time, joint_path_length(path), context.collision_checks - start_checks, self.name, "success")
            a_nodes, b_nodes = b_nodes, a_nodes
            a_parents, b_parents = b_parents, a_parents
            a_is_start = not a_is_start
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
