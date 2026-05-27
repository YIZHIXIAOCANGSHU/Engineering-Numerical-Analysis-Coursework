from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from arm_planning.core.types import PlanResult
from arm_planning.sim.mujoco_world import MujocoWorld


class Planner(ABC):
    name: str

    @abstractmethod
    def plan(self, q_start: np.ndarray, q_goal: np.ndarray, context: MujocoWorld, rng: np.random.Generator) -> PlanResult:
        raise NotImplementedError
