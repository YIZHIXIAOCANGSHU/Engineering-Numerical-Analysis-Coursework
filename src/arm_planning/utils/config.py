from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from arm_planning.core.types import ObstacleSpec, ProjectConfig, RobotSpec, SceneSpec
from arm_planning.experiments.demo_scene import mixed_maze_scene


ROOT = Path(__file__).resolve().parents[3]


def project_root() -> Path:
    return ROOT


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_optional_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return load_yaml(config_path)


def _obstacle_from_dict(data: dict[str, Any]) -> ObstacleSpec:
    return ObstacleSpec(
        type=str(data["type"]),
        name=str(data["name"]),
        position=np.asarray(data["position"], dtype=float),
        size=np.asarray(data["size"], dtype=float) if "size" in data else None,
        radius=float(data["radius"]) if "radius" in data else None,
        height=float(data["height"]) if "height" in data else None,
        quat_wxyz=np.asarray(data["quat_wxyz"], dtype=float) if "quat_wxyz" in data else None,
    )


def _auto_ee_start(robot: RobotSpec) -> np.ndarray:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(resolve_path(robot.model_xml)))
    data = mujoco.MjData(model)
    for name, value in zip(robot.joint_names, robot.q_start):
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = float(value)
    mujoco.mj_forward(model, data)
    return data.site(model.site(robot.ee_site).id).xpos.copy()


def _scene_from_dict(item: dict[str, Any], global_seed: int, robot: RobotSpec) -> SceneSpec:
    scene = SceneSpec(
        id=str(item["id"]),
        seed=int(item.get("seed", global_seed)),
        q_start=robot.q_start.copy(),
        target_position=np.asarray(item["target_position"], dtype=float),
        obstacles=[_obstacle_from_dict(obs) for obs in item.get("obstacles", [])],
        target_quat_wxyz=np.asarray(item["target_quat_wxyz"], dtype=float) if "target_quat_wxyz" in item else None,
    )
    generator = item.get("generator", {})
    if generator.get("type") == "mixed_maze":
        if generator.get("ee_start") == "auto":
            ee_start = _auto_ee_start(robot)
        else:
            ee_start = np.asarray(generator.get("ee_start", robot.q_start[:3]), dtype=float)
        scene = mixed_maze_scene(scene, ee_start, generator)
    return scene


def load_project_config(
    scenes_path: str | Path = "configs/scenes.yaml",
    experiments_path: str | Path = "configs/experiments.yaml",
    config_path: str | Path | None = None,
) -> ProjectConfig:
    root = project_root()
    scenes_raw = load_yaml(root / scenes_path)
    experiments_raw = load_yaml(root / experiments_path)
    overlay_raw = _load_optional_yaml(config_path)
    if "global_seed" in overlay_raw:
        scenes_raw["global_seed"] = overlay_raw["global_seed"]
    if "robot" in overlay_raw:
        scenes_raw["robot"] = {**scenes_raw.get("robot", {}), **overlay_raw["robot"]}
    for section in ("experiment", "ik", "planners", "trajectory", "rerun"):
        if section in overlay_raw:
            experiments_raw[section] = {**experiments_raw.get(section, {}), **overlay_raw[section]}
    robot_raw = scenes_raw["robot"]
    robot = RobotSpec(
        name=str(robot_raw["name"]),
        model_xml=str(robot_raw["model_xml"]),
        ee_site=str(robot_raw["ee_site"]),
        joint_names=list(robot_raw["joint_names"]),
        q_start=np.asarray(robot_raw["q_start"], dtype=float),
    )
    scenes = []
    for item in scenes_raw["scenes"]:
        scenes.append(_scene_from_dict(item, int(scenes_raw["global_seed"]), robot))
    return ProjectConfig(
        global_seed=int(scenes_raw["global_seed"]),
        robot=robot,
        scenes=scenes,
        experiment=experiments_raw.get("experiment", {}),
        ik=experiments_raw.get("ik", {}),
        planners=experiments_raw.get("planners", {}),
        trajectory=experiments_raw.get("trajectory", {}),
        rerun=experiments_raw.get("rerun", {}),
    )


def ensure_output_dirs() -> None:
    for rel in ["results/data", "results/recordings", "results/figures", "images/generated", "assets/models", "third_party"]:
        (project_root() / rel).mkdir(parents=True, exist_ok=True)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p
