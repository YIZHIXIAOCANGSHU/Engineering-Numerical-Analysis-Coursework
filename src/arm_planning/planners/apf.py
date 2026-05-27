from __future__ import annotations

import time

import numpy as np

from arm_planning.core.math_utils import clip_to_limits, joint_path_length
from arm_planning.core.types import PlanResult
from arm_planning.planners.base import Planner
from arm_planning.sim.mujoco_world import MujocoWorld


class APFPlanner(Planner):
    name = "apf"

    def __init__(self, params: dict):
        self.max_iterations = int(params.get("max_iterations", 700))
        self.step_size = float(params.get("apf_step_size", 0.045))
        self.goal_tolerance = float(params.get("goal_tolerance", 0.22))
        self.k_att = float(params.get("apf_attractive_gain", 1.0))
        self.k_rep = float(params.get("apf_repulsive_gain", 0.018))
        self.rep_radius = float(params.get("apf_repulsive_radius", 0.35))

    def _repulsive_gradient(self, q: np.ndarray, context: MujocoWorld) -> np.ndarray:
        base_dist = context.min_obstacle_distance(q)
        if not np.isfinite(base_dist) or base_dist >= self.rep_radius:
            return np.zeros_like(q)
        eps = 1e-3
        grad_d = np.zeros_like(q)
        for i in range(len(q)):
            qp = q.copy(); qp[i] += eps
            qm = q.copy(); qm[i] -= eps
            dp = context.min_obstacle_distance(clip_to_limits(qp, context.lower_limits, context.upper_limits))
            dm = context.min_obstacle_distance(clip_to_limits(qm, context.lower_limits, context.upper_limits))
            grad_d[i] = (dp - dm) / (2.0 * eps)
        scale = self.k_rep * (1.0 / max(base_dist, 1e-3) - 1.0 / self.rep_radius) / max(base_dist, 1e-3) ** 2
        return scale * grad_d

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray, context: MujocoWorld, rng: np.random.Generator) -> PlanResult:
        del rng
        start_time = time.perf_counter()
        start_checks = context.collision_checks
        if not context.is_state_valid(q_start) or not context.is_state_valid(q_goal):
            return PlanResult(False, [], time.perf_counter() - start_time, 0.0, context.collision_checks - start_checks, self.name, "invalid start or goal")
        q = q_start.copy()
        path = [q.copy()]
        stagnant = 0
        prev_goal_dist = float(np.linalg.norm(q_goal - q))
        for _ in range(self.max_iterations):
            to_goal = q_goal - q
            if np.linalg.norm(to_goal) <= self.goal_tolerance and context.is_edge_valid(q, q_goal):
                path.append(q_goal.copy())
                return PlanResult(True, path, time.perf_counter() - start_time, joint_path_length(path), context.collision_checks - start_checks, self.name, "success")
            direction = self.k_att * to_goal + self._repulsive_gradient(q, context)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                return PlanResult(False, path, time.perf_counter() - start_time, joint_path_length(path), context.collision_checks - start_checks, self.name, "local minimum", {"failure_q": q.copy()})
            q_next = clip_to_limits(q + direction / norm * self.step_size, context.lower_limits, context.upper_limits)
            if not context.is_edge_valid(q, q_next):
                # Deterministic sidestep based on the largest free joint-space axis.
                moved = False
                for axis in np.argsort(-np.abs(to_goal)):
                    for sign in (1.0, -1.0):
                        candidate = q.copy()
                        candidate[int(axis)] += sign * self.step_size
                        candidate = clip_to_limits(candidate, context.lower_limits, context.upper_limits)
                        if context.is_edge_valid(q, candidate):
                            q_next = candidate
                            moved = True
                            break
                    if moved:
                        break
                if not moved:
                    return PlanResult(False, path, time.perf_counter() - start_time, joint_path_length(path), context.collision_checks - start_checks, self.name, "blocked by obstacle", {"failure_q": q.copy()})
            q = q_next
            path.append(q.copy())
            goal_dist = float(np.linalg.norm(q_goal - q))
            stagnant = stagnant + 1 if goal_dist >= prev_goal_dist - 1e-4 else 0
            prev_goal_dist = goal_dist
            if stagnant > 80:
                return PlanResult(False, path, time.perf_counter() - start_time, joint_path_length(path), context.collision_checks - start_checks, self.name, "local minimum", {"failure_q": q.copy()})
        return PlanResult(False, path, time.perf_counter() - start_time, joint_path_length(path), context.collision_checks - start_checks, self.name, "max iterations reached", {"failure_q": q.copy()})
