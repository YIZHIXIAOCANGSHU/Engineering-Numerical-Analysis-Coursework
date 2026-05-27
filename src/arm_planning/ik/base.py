from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from arm_planning.core.types import IKResult
from arm_planning.sim.mujoco_world import MujocoWorld


class IKSolver(ABC):
    name: str

    @abstractmethod
    def solve(
        self,
        target_position: np.ndarray,
        q_seed: np.ndarray,
        context: MujocoWorld,
        target_quat_wxyz: np.ndarray | None = None,
    ) -> IKResult:
        raise NotImplementedError
