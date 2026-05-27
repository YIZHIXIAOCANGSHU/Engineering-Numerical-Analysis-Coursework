from __future__ import annotations

import heapq
import time

import numpy as np

from arm_planning.core.math_utils import joint_path_length
from arm_planning.core.types import PlanResult
from arm_planning.planners.base import Planner
from arm_planning.planners.rrt import _sample
from arm_planning.sim.mujoco_world import MujocoWorld


class PRMPlanner(Planner):
    name = "prm"

    def __init__(self, params: dict):
        self.samples = int(params.get("prm_samples", 170))
        self.neighbors = int(params.get("prm_neighbors", 10))

    def _search(self, graph: list[list[tuple[int, float]]], start: int, goal: int) -> list[int] | None:
        pq = [(0.0, start)]
        prev = {start: -1}
        dist = {start: 0.0}
        while pq:
            cost, node = heapq.heappop(pq)
            if node == goal:
                ids = []
                while node != -1:
                    ids.append(node)
                    node = prev[node]
                return list(reversed(ids))
            if cost > dist[node]:
                continue
            for nxt, w in graph[node]:
                nd = cost + w
                if nd < dist.get(nxt, float("inf")):
                    dist[nxt] = nd
                    prev[nxt] = node
                    heapq.heappush(pq, (nd, nxt))
        return None

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray, context: MujocoWorld, rng: np.random.Generator) -> PlanResult:
        start_time = time.perf_counter()
        start_checks = context.collision_checks
        if not context.is_state_valid(q_start) or not context.is_state_valid(q_goal):
            return PlanResult(False, [], time.perf_counter() - start_time, 0.0, context.collision_checks - start_checks, self.name, "invalid start or goal")
        nodes = [q_start.copy(), q_goal.copy()]
        attempts = 0
        while len(nodes) < self.samples + 2 and attempts < self.samples * 20:
            attempts += 1
            q = _sample(context, rng)
            if context.is_state_valid(q):
                nodes.append(q)
        graph: list[list[tuple[int, float]]] = [[] for _ in nodes]
        arr = np.asarray(nodes)
        for i, q in enumerate(nodes):
            dists = np.linalg.norm(arr - q, axis=1)
            order = np.argsort(dists)[1 : self.neighbors + 1]
            for j in order:
                if j <= i:
                    continue
                if context.is_edge_valid(nodes[i], nodes[int(j)]):
                    w = float(dists[int(j)])
                    graph[i].append((int(j), w))
                    graph[int(j)].append((i, w))
        ids = self._search(graph, 0, 1)
        if ids is None:
            dists = np.linalg.norm(np.asarray(nodes) - q_goal, axis=1)
            best_q = nodes[int(np.argmin(dists))].copy()
            return PlanResult(
                False,
                [],
                time.perf_counter() - start_time,
                0.0,
                context.collision_checks - start_checks,
                self.name,
                "no roadmap path",
                {"failure_q": best_q},
            )
        path = [nodes[i] for i in ids]
        return PlanResult(True, path, time.perf_counter() - start_time, joint_path_length(path), context.collision_checks - start_checks, self.name, "success")
