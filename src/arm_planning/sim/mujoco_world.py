from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import mujoco
import numpy as np

from arm_planning.core.math_utils import interpolate_joint_path
from arm_planning.core.types import ObstacleSpec, RobotSpec, SceneSpec
from arm_planning.utils.config import resolve_path


GROUND_PENETRATION_TOLERANCE = -1e-4


class MujocoWorld:
    """Small MuJoCo adapter used by IK, planners, metrics, and demos."""

    def __init__(self, robot: RobotSpec, edge_resolution: float = 0.06):
        self.robot_spec = robot
        self.edge_resolution = float(edge_resolution)
        self.model_path = resolve_path(robot.model_xml)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.ee_site_id = self.model.site(robot.ee_site).id
        self.joint_names = list(robot.joint_names)
        self.joint_ids = [self.model.joint(name).id for name in self.joint_names]
        self.qpos_addrs = np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=int)
        self.dof_addrs = np.array([self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=int)
        ranges = np.array([self.model.joint(name).range for name in self.joint_names], dtype=float)
        self.lower_limits = ranges[:, 0]
        self.upper_limits = ranges[:, 1]
        self.scene: SceneSpec | None = None
        self.obstacle_specs: list[ObstacleSpec] = []
        self.obstacle_geom_ids: list[int] = []
        self.ground_geom_ids: list[int] = self._collect_ground_geom_ids()
        self.robot_geom_ids: list[int] = self._collect_robot_geom_ids()
        self.collision_checks = 0
        self.reset(robot.q_start)

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def _collect_robot_geom_ids(self) -> list[int]:
        ids: list[int] = []
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith("robot_"):
                ids.append(geom_id)
        if ids:
            return ids
        fallback_ids: list[int] = []
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name == "floor" or name.startswith("obs_"):
                continue
            if self.model.geom_contype[geom_id] or self.model.geom_conaffinity[geom_id]:
                fallback_ids.append(geom_id)
        return fallback_ids

    def _collect_ground_geom_ids(self) -> list[int]:
        ids: list[int] = []
        for geom_id in range(self.model.ngeom):
            name = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").lower()
            geom_type = int(self.model.geom_type[geom_id])
            if name in {"floor", "ground", "plane"} or "floor" in name or "ground" in name or geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
                ids.append(geom_id)
        return ids

    def reset(self, q: np.ndarray | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if q is not None:
            self.set_qpos(q)
        else:
            mujoco.mj_forward(self.model, self.data)
        self.collision_checks = 0

    def load_scene(self, scene: SceneSpec) -> None:
        self.scene = scene
        self.obstacle_specs = list(scene.obstacles)
        self.obstacle_geom_ids = []
        for obs in self.obstacle_specs:
            try:
                geom_id = self.model.geom(obs.name).id
                self.obstacle_geom_ids.append(int(geom_id))
            except KeyError:
                # The fallback model is static; generated scene XML should include obstacles.
                pass
        self.robot_geom_ids = [geom_id for geom_id in self._collect_robot_geom_ids() if geom_id not in self.obstacle_geom_ids]
        self.reset(scene.q_start)

    def set_qpos(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        self.data.qpos[self.qpos_addrs] = q
        self.data.qvel[self.dof_addrs] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_qpos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_addrs].copy()

    def get_ee_position(self) -> np.ndarray:
        return self.data.site(self.ee_site_id).xpos.copy()

    def get_ee_quat(self) -> np.ndarray:
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, self.data.site(self.ee_site_id).xmat.copy())
        if quat[0] < 0.0:
            quat *= -1.0
        return quat

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        self.set_qpos(q)
        return self.get_ee_position()

    def forward_quat(self, q: np.ndarray) -> np.ndarray:
        self.set_qpos(q)
        return self.get_ee_quat()

    def compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        self.set_qpos(q)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee_site_id)
        return jacp[:, self.dof_addrs].copy()

    def compute_pose_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.set_qpos(q)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        return jacp[:, self.dof_addrs].copy(), jacr[:, self.dof_addrs].copy()

    def condition_number(self, q: np.ndarray) -> float:
        jac = self.compute_jacobian(q)
        try:
            return float(np.linalg.cond(jac))
        except np.linalg.LinAlgError:
            return float("inf")

    def is_within_limits(self, q: np.ndarray) -> bool:
        q = np.asarray(q, dtype=float)
        return bool(np.all(q >= self.lower_limits) and np.all(q <= self.upper_limits))

    def is_state_valid(self, q: np.ndarray) -> bool:
        self.collision_checks += 1
        if not self.is_within_limits(q):
            return False
        self.set_qpos(q)
        if self.ground_clearance_current() < GROUND_PENETRATION_TOLERANCE:
            return False
        if self.robot_geom_ids and self.obstacle_geom_ids:
            for robot_gid in self.robot_geom_ids:
                for obstacle_gid in self.obstacle_geom_ids:
                    dist = self.geom_distance(robot_gid, obstacle_gid, distmax=1.0)
                    if dist < 0.0:
                        return False
        else:
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom[j])) or "" for j in (0, 1)]
                if any(name.startswith("robot_") for name in names) and any(name.startswith("obs_") or name in {obs.name for obs in self.obstacle_specs} for name in names):
                    return False
        return True

    def is_edge_valid(self, q1: np.ndarray, q2: np.ndarray, resolution: float | None = None) -> bool:
        resolution = self.edge_resolution if resolution is None else float(resolution)
        for q in interpolate_joint_path(q1, q2, resolution):
            if not self.is_state_valid(q):
                return False
        return True

    def geom_distance(self, geom_id_1: int, geom_id_2: int, distmax: float = 10.0) -> float:
        fromto = np.zeros(6, dtype=float)
        return float(mujoco.mj_geomDistance(self.model, self.data, int(geom_id_1), int(geom_id_2), distmax, fromto))

    def ground_clearance_current(self) -> float:
        if not self.robot_geom_ids or not self.ground_geom_ids:
            return float("inf")
        best = float("inf")
        for robot_gid in self.robot_geom_ids:
            for ground_gid in self.ground_geom_ids:
                best = min(best, self.geom_distance(robot_gid, ground_gid, distmax=10.0))
        return float(best)

    def ground_clearance(self, q: np.ndarray) -> float:
        self.set_qpos(q)
        return self.ground_clearance_current()

    def min_ground_clearance(self, path: list[np.ndarray]) -> float:
        if not path:
            return float("nan")
        return float(min(self.ground_clearance(q) for q in path))

    def min_obstacle_distance(self, q: np.ndarray) -> float:
        self.set_qpos(q)
        if not self.robot_geom_ids or not self.obstacle_geom_ids:
            return float("inf")
        best = float("inf")
        for robot_gid in self.robot_geom_ids:
            for obstacle_gid in self.obstacle_geom_ids:
                best = min(best, self.geom_distance(robot_gid, obstacle_gid, distmax=10.0))
        return float(best)

    def sample_end_effector_path(self, path: list[np.ndarray]) -> np.ndarray:
        points = []
        for q in path:
            points.append(self.forward_kinematics(q))
        return np.asarray(points, dtype=float)

    def sample_end_effector_quats(self, path: list[np.ndarray]) -> np.ndarray:
        quats = []
        for q in path:
            quats.append(self.forward_quat(q))
        return np.asarray(quats, dtype=float)

    def play_trajectory(
        self,
        path: list[np.ndarray],
        seconds_per_waypoint: float = 0.04,
        loop: bool = True,
        ee_points: np.ndarray | None = None,
        wait_after_playback: bool = False,
    ) -> None:
        import mujoco.viewer

        display_path = self._display_path(path, min_samples=80)
        display_ee_points = self.sample_end_effector_path(display_path)
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self._update_viewer_overlays(viewer, display_ee_points)
            while viewer.is_running():
                for q in display_path:
                    if not viewer.is_running():
                        break
                    self.set_qpos(q)
                    self._update_viewer_overlays(viewer, display_ee_points)
                    viewer.sync()
                    time.sleep(seconds_per_waypoint)
                if not loop:
                    break
            while wait_after_playback and viewer.is_running():
                self._update_viewer_overlays(viewer, display_ee_points)
                viewer.sync()
                time.sleep(seconds_per_waypoint)

    @staticmethod
    def _display_path(path: list[np.ndarray], min_samples: int) -> list[np.ndarray]:
        if len(path) < 2:
            return [q.copy() for q in path]
        arr = np.asarray(path, dtype=float)
        distances = np.zeros(len(arr), dtype=float)
        for idx in range(1, len(arr)):
            distances[idx] = distances[idx - 1] + float(np.linalg.norm(arr[idx] - arr[idx - 1]))
        if distances[-1] <= 1e-12:
            return [arr[0].copy() for _ in range(min_samples)]
        samples = max(min_samples, len(path))
        support = distances / distances[-1]
        return [np.array([np.interp(t, support, arr[:, joint]) for joint in range(arr.shape[1])], dtype=float) for t in np.linspace(0.0, 1.0, samples)]

    def _update_viewer_overlays(self, viewer, ee_points: np.ndarray | None) -> None:
        user_scn = getattr(viewer, "user_scn", None)
        if user_scn is None:
            return
        user_scn.ngeom = 0
        ee_pos = self.get_ee_position()
        self._add_viewer_sphere(user_scn, ee_pos, radius=0.024, rgba=(0.1, 0.55, 1.0, 1.0))
        if self.scene is not None:
            self._add_viewer_sphere(user_scn, self.scene.target_position, radius=0.045, rgba=(0.0, 0.9, 0.2, 0.9))
        if ee_points is not None and len(ee_points) >= 2:
            for start, end in zip(ee_points[:-1], ee_points[1:]):
                self._add_viewer_line(user_scn, start, end, rgba=(0.1, 0.55, 1.0, 0.75))

    @staticmethod
    def _add_viewer_sphere(user_scn, pos: np.ndarray, radius: float, rgba: tuple[float, float, float, float]) -> None:
        if user_scn.ngeom >= user_scn.maxgeom:
            return
        geom = user_scn.geoms[user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.asarray([radius, radius, radius], dtype=float),
            np.asarray(pos, dtype=float),
            np.eye(3, dtype=float).ravel(),
            np.asarray(rgba, dtype=np.float32),
        )
        user_scn.ngeom += 1

    @staticmethod
    def _add_viewer_line(user_scn, start: np.ndarray, end: np.ndarray, rgba: tuple[float, float, float, float]) -> None:
        if user_scn.ngeom >= user_scn.maxgeom:
            return
        geom = user_scn.geoms[user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            np.eye(3, dtype=float).ravel(),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 4.0, np.asarray(start, dtype=float), np.asarray(end, dtype=float))
        user_scn.ngeom += 1


def build_scene_xml(robot: RobotSpec, scene: SceneSpec, output_path: str | Path | None = None) -> Path:
    """Create a scene XML that includes the static obstacle geoms.

    The fallback model has no include mechanism for dynamic obstacle creation at runtime, so
    experiments generate a deterministic XML per scene.
    """

    base_path = resolve_path(robot.model_xml)
    text = base_path.read_text(encoding="utf-8")
    insert = []
    obstacle_attrs = 'rgba="0.9 0.25 0.15 0.75" contype="1" conaffinity="1"'
    target_pos = " ".join(f"{x:.8g}" for x in scene.target_position)
    insert.append(f'    <geom name="target_marker" type="sphere" pos="{target_pos}" size="0.045" rgba="0 0.9 0.2 0.55" contype="0" conaffinity="0"/>')
    insert.append(f'    <site name="target_site" pos="{target_pos}" size="0.055" rgba="0 1 0.15 1"/>')
    for obs in scene.obstacles:
        pos = " ".join(f"{x:.8g}" for x in obs.position)
        quat_attr = ""
        if obs.quat_wxyz is not None:
            quat = np.asarray(obs.quat_wxyz, dtype=float)
            quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
            quat_attr = ' quat="' + " ".join(f"{x:.8g}" for x in quat) + '"'
        if obs.type == "box":
            assert obs.size is not None
            half = np.asarray(obs.size, dtype=float) / 2.0
            size = " ".join(f"{x:.8g}" for x in half)
            insert.append(f'    <geom name="{obs.name}" type="box" pos="{pos}" size="{size}"{quat_attr} {obstacle_attrs}/>')
        elif obs.type == "sphere":
            assert obs.radius is not None
            insert.append(f'    <geom name="{obs.name}" type="sphere" pos="{pos}" size="{float(obs.radius):.8g}"{quat_attr} {obstacle_attrs}/>')
        elif obs.type == "cylinder":
            assert obs.radius is not None and obs.height is not None
            insert.append(f'    <geom name="{obs.name}" type="cylinder" pos="{pos}" size="{float(obs.radius):.8g} {float(obs.height) / 2.0:.8g}"{quat_attr} {obstacle_attrs}/>')
        else:
            raise ValueError(f"Unsupported obstacle type: {obs.type}")
    marker = "  </worldbody>"
    if marker not in text:
        raise ValueError(f"Could not find worldbody close tag in {base_path}")
    text = text.replace(marker, "\n".join(insert) + "\n" + marker, 1)
    if output_path is None:
        if "<include" in text:
            output_path = base_path.with_name(f"{base_path.stem}_{scene.id}.generated.xml")
        else:
            output_path = resolve_path("assets/models") / f"{scene.id}.xml"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.stem}.{uuid4().hex}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(output_path)
    return output_path


def world_for_scene(robot: RobotSpec, scene: SceneSpec, edge_resolution: float = 0.06) -> MujocoWorld:
    scene_xml = build_scene_xml(robot, scene)
    scene_robot = RobotSpec(
        name=robot.name,
        model_xml=str(scene_xml),
        ee_site=robot.ee_site,
        joint_names=robot.joint_names,
        q_start=robot.q_start,
    )
    world = MujocoWorld(scene_robot, edge_resolution=edge_resolution)
    world.load_scene(scene)
    return world
