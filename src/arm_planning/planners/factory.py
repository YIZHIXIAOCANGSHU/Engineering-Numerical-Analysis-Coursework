from __future__ import annotations

from arm_planning.planners.apf import APFPlanner
from arm_planning.planners.base import Planner
from arm_planning.planners.prm import PRMPlanner
from arm_planning.planners.rrt import RRTPlanner
from arm_planning.planners.rrt_connect import RRTConnectPlanner


def create_planner(name: str, params: dict) -> Planner:
    if name == "rrt":
        return RRTPlanner(params)
    if name == "rrt_connect":
        return RRTConnectPlanner(params)
    if name == "prm":
        return PRMPlanner(params)
    if name == "apf":
        return APFPlanner(params)
    raise ValueError(f"Unknown planner: {name}")
