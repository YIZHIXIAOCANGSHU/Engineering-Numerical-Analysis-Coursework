from __future__ import annotations

import numpy as np
import pandas as pd

from arm_planning.analysis.metrics import compute_trial_metrics, trajectory_kinematic_metrics
from arm_planning.experiments.demo_scene import random_blocking_scene
from arm_planning.experiments.failure import failure_category
from arm_planning.experiments.result_store import ResultStore, array_to_string
from arm_planning.ik.solvers import create_ik_solver, orientation_error_vector
from arm_planning.planners.factory import create_planner
from arm_planning.sim.mujoco_world import GROUND_PENETRATION_TOLERANCE, world_for_scene
from arm_planning.trajectory.smoothing import process_path
from arm_planning.utils.config import ensure_output_dirs, load_project_config
from arm_planning.utils.reproducibility import create_rng, make_trial_seed, set_global_seed
from arm_planning.viz.rerun_logger import RerunLogger


def _path_is_valid(path: list[np.ndarray], world) -> bool:
    if not path:
        return False
    if len(path) == 1:
        return bool(world.is_state_valid(path[0]))
    return all(world.is_edge_valid(path[i], path[i + 1]) for i in range(len(path) - 1))


def _validation_message(world, q_start: np.ndarray, q_goal: np.ndarray) -> str:
    if world.ground_clearance(q_start) < GROUND_PENETRATION_TOLERANCE or world.ground_clearance(q_goal) < GROUND_PENETRATION_TOLERANCE:
        return "ground penetration at start or goal"
    if not world.is_state_valid(q_start) or not world.is_state_valid(q_goal):
        return "start or IK goal is blocked"
    return ""


def _selected_scenes(cfg, scene_id: str | None = None):
    if scene_id is None:
        return list(enumerate(cfg.scenes))
    for idx, scene in enumerate(cfg.scenes):
        if scene.id == scene_id:
            return [(idx, scene)]
    known = ", ".join(scene.id for scene in cfg.scenes)
    raise ValueError(f"Unknown scene: {scene_id}. Available scenes: {known}")


def run_ik_suite(store: ResultStore, trials: int | None = None, config_path: str | None = None, scene_id: str | None = None) -> dict[tuple[str, str], np.ndarray]:
    cfg = load_project_config(config_path=config_path)
    ensure_output_dirs()
    set_global_seed(cfg.global_seed)
    q_goals: dict[tuple[str, str], np.ndarray] = {}
    for scene_idx, scene in _selected_scenes(cfg, scene_id):
        for method_idx, method in enumerate(cfg.ik.get("methods", [])):
            repeats = 1 if trials is None else max(1, int(trials))
            best_q = None
            best_err = float("inf")
            for trial_id in range(repeats):
                world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
                seed = make_trial_seed(cfg.global_seed, scene_idx, method_idx, trial_id)
                rng = create_rng(seed)
                q_seed = scene.q_start + rng.normal(0.0, 0.02, size=len(scene.q_start))
                result = create_ik_solver(method, cfg.ik).solve(scene.target_position, q_seed, world, scene.target_quat_wxyz)
                if result.position_error < best_err:
                    best_err = result.position_error
                    best_q = result.q.copy()
                store.save_ik_result({
                    "trial_id": trial_id,
                    "global_seed": cfg.global_seed,
                    "scene_id": scene.id,
                    "scene_seed": scene.seed,
                    "ik_method": method,
                    "trial_seed": seed,
                    "success": result.success,
                    "solve_time": result.solve_time,
                    "iterations": result.iterations,
                    "position_error": result.position_error,
                    "orientation_error": result.orientation_error,
                    "condition_number": result.condition_number,
                    "q_solution": array_to_string(result.q),
                    "message": result.message,
                })
                store.save_reproducibility_row({
                    "trial_id": trial_id,
                    "global_seed": cfg.global_seed,
                    "scene_id": scene.id,
                    "scene_seed": scene.seed,
                    "algorithm_type": "ik",
                    "algorithm": method,
                    "trial_seed": seed,
                    "q_start": array_to_string(scene.q_start),
                    "q_goal": array_to_string(result.q),
                    "target_position": array_to_string(scene.target_position),
                    "target_quat_wxyz": array_to_string(scene.target_quat_wxyz) if scene.target_quat_wxyz is not None else "",
                    "success": result.success,
                })
            if best_q is not None:
                q_goals[(scene.id, method)] = best_q
    return q_goals


def solve_goal_for_scene(scene, cfg, ik_method: str | None = None, world=None) -> np.ndarray:
    world = world or world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
    preferred = ik_method or ("scipy_baseline" if "scipy_baseline" in cfg.ik.get("methods", []) else cfg.ik.get("methods", ["dls"])[0])
    result = create_ik_solver(preferred, cfg.ik).solve(scene.target_position, scene.q_start, world, scene.target_quat_wxyz)
    if not result.success:
        raise RuntimeError(f"IK failed for {scene.id}: {result.message}, error={result.position_error}")
    return result.q


def _demo_obstacle_params(obstacle_params: dict, attempt: int, default_seed: int) -> dict:
    params = dict(obstacle_params)
    if params.get("enabled", False):
        params["seed"] = int(obstacle_params.get("seed", default_seed)) + attempt
    return params


def demo_plan(scene, cfg, ik_method: str | None, planner_name: str):
    obstacle_params = cfg.experiment.get("random_obstacles", {})
    edge_resolution = cfg.experiment.get("edge_resolution", 0.06)
    base_world = world_for_scene(cfg.robot, scene, edge_resolution)
    ee_start = base_world.forward_kinematics(scene.q_start)
    max_attempts = max(1, int(obstacle_params.get("max_attempts", 1))) if obstacle_params.get("enabled", False) else 1
    last_message = ""
    for attempt in range(max_attempts):
        candidate = random_blocking_scene(scene, ee_start, _demo_obstacle_params(obstacle_params, attempt, cfg.global_seed))
        world = world_for_scene(cfg.robot, candidate, edge_resolution)
        try:
            q_goal = solve_goal_for_scene(candidate, cfg, ik_method=ik_method, world=world)
        except RuntimeError as exc:
            last_message = str(exc)
            continue
        if world.is_state_valid(candidate.q_start) and world.is_state_valid(q_goal):
            planner = create_planner(planner_name, cfg.planners)
            result = planner.plan(candidate.q_start, q_goal, world, create_rng(cfg.global_seed + attempt))
            if result.success:
                return candidate, world, q_goal, result, attempt + 1
            last_message = result.message
        else:
            last_message = "start or goal is blocked"
    raise RuntimeError(f"Demo planning failed after {max_attempts} random scene attempts: {last_message}")


def validate_scenes(config_path: str | None = None, scene_id: str | None = None) -> None:
    cfg = load_project_config(config_path=config_path)
    for _, scene in _selected_scenes(cfg, scene_id):
        world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
        if not world.is_state_valid(scene.q_start):
            raise RuntimeError(f"Invalid q_start for {scene.id}")
        q_goal = solve_goal_for_scene(scene, cfg)
        if not world.is_state_valid(q_goal):
            raise RuntimeError(f"Invalid IK goal for {scene.id}")


def trajectory_rows(
    scene_id: str,
    planner: str,
    trial_id: int,
    path: list[np.ndarray],
    ee_points: np.ndarray,
    world,
    target_position: np.ndarray,
    target_quat_wxyz: np.ndarray | None = None,
) -> list[dict]:
    rows = []
    q_arr = np.asarray(path, dtype=float) if path else np.empty((0, 0))
    ee_arr = np.asarray(ee_points, dtype=float)
    ee_quats = world.sample_end_effector_quats(path) if path else np.empty((0, 4))
    cumulative_task = np.zeros(len(path), dtype=float)
    if len(ee_arr) >= 2:
        cumulative_task[1:] = np.cumsum(np.linalg.norm(np.diff(ee_arr, axis=0), axis=1))
    for i, q in enumerate(path):
        point = ee_points[i] if i < len(ee_points) else np.full(3, np.nan)
        quat = ee_quats[i] if i < len(ee_quats) else np.full(4, np.nan)
        ee_orientation_error = (
            float(np.linalg.norm(orientation_error_vector(quat, target_quat_wxyz)))
            if target_quat_wxyz is not None and np.all(np.isfinite(quat))
            else 0.0
        )
        if len(q_arr) >= 2 and i >= 1:
            joint_speed_norm = float(np.linalg.norm(q_arr[i] - q_arr[i - 1]))
        else:
            joint_speed_norm = 0.0
        if len(q_arr) >= 3 and i >= 2:
            joint_acc_norm = float(np.linalg.norm(q_arr[i] - 2.0 * q_arr[i - 1] + q_arr[i - 2]))
        else:
            joint_acc_norm = 0.0
        if len(q_arr) >= 4 and i >= 3:
            joint_jerk_norm = float(np.linalg.norm(q_arr[i] - 3.0 * q_arr[i - 1] + 3.0 * q_arr[i - 2] - q_arr[i - 3]))
        else:
            joint_jerk_norm = 0.0
        rows.append({
            "scene_id": scene_id,
            "planner": planner,
            "trial_id": trial_id,
            "sample_index": i,
            "q": array_to_string(q),
            "ee_x": point[0],
            "ee_y": point[1],
            "ee_z": point[2],
            "ee_qw": quat[0],
            "ee_qx": quat[1],
            "ee_qy": quat[2],
            "ee_qz": quat[3],
            "ee_error": float(np.linalg.norm(point - target_position)) if np.all(np.isfinite(point)) else float("nan"),
            "ee_orientation_error": ee_orientation_error,
            "cumulative_task_length": float(cumulative_task[i]) if i < len(cumulative_task) else float("nan"),
            "min_obstacle_distance": world.min_obstacle_distance(q),
            "min_ground_clearance": world.ground_clearance(q),
            "joint_speed_norm": joint_speed_norm,
            "joint_acc_norm": joint_acc_norm,
            "joint_jerk_norm": joint_jerk_norm,
        })
    return rows


def _snapshot_q(result, processed_path: list[np.ndarray]) -> np.ndarray | None:
    if result is None:
        return None
    failure_q = result.metadata.get("failure_q") if getattr(result, "metadata", None) is not None else None
    if failure_q is not None:
        return np.asarray(failure_q, dtype=float)
    if processed_path:
        return np.asarray(processed_path[-1], dtype=float)
    if result.path:
        return np.asarray(result.path[-1], dtype=float)
    return None


def _failure_snapshot_row(
    scene,
    planner_name: str,
    trial_id: int,
    seed: int,
    category: str,
    category_cn: str,
    message: str,
    result,
    processed_path: list[np.ndarray],
    world,
) -> dict | None:
    q_snapshot = _snapshot_q(result, processed_path)
    if q_snapshot is None:
        return None
    ee_pos = world.forward_kinematics(q_snapshot)
    return {
        "trial_id": trial_id,
        "scene_id": scene.id,
        "scene_seed": scene.seed,
        "planner": planner_name,
        "trial_seed": seed,
        "failure_category": category,
        "failure_category_cn": category_cn,
        "message": message,
        "q_snapshot": array_to_string(q_snapshot),
        "ee_x": float(ee_pos[0]),
        "ee_y": float(ee_pos[1]),
        "ee_z": float(ee_pos[2]),
        "ee_error": float(np.linalg.norm(ee_pos - scene.target_position)),
        "min_obstacle_distance": world.min_obstacle_distance(q_snapshot),
        "min_ground_clearance": world.ground_clearance(q_snapshot),
    }


def run_planner_suite(store: ResultStore, trials: int | None = None, log_rerun: bool = False, config_path: str | None = None, scene_id: str | None = None) -> None:
    cfg = load_project_config(config_path=config_path)
    ensure_output_dirs()
    set_global_seed(cfg.global_seed)
    repeats = int(trials if trials is not None else cfg.experiment.get("trials", 30))
    logger = RerunLogger(
        enabled=bool(log_rerun and cfg.rerun.get("enabled", True)),
        recording_path=cfg.rerun.get("recording_path", "results/recordings/arm_planning.rrd"),
        spawn=cfg.rerun.get("spawn", False),
    )
    for scene_idx, scene in _selected_scenes(cfg, scene_id):
        q_goal = solve_goal_for_scene(scene, cfg)
        if log_rerun:
            logger.log_scene(scene)
        for planner_idx, planner_name in enumerate(cfg.planners.get("methods", [])):
            for trial_id in range(repeats):
                world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
                planner = create_planner(planner_name, cfg.planners)
                seed = make_trial_seed(cfg.global_seed, scene_idx, planner_idx, trial_id)
                rng = create_rng(seed)
                result = planner.plan(scene.q_start, q_goal, world, rng)
                processed_path = []
                if result.success:
                    processed_path = process_path(
                        result.path,
                        world,
                        cfg.trajectory.get("shortcut_iterations", 80),
                        cfg.trajectory.get("samples", 80),
                        rng,
                    )
                    if not _path_is_valid(processed_path, world):
                        result.success = False
                        result.message = "postprocess invalid: collision or ground penetration"
                elif result.path:
                    processed_path = result.path
                metrics = compute_trial_metrics(processed_path, world, scene.target_position, result.planning_time, result.collision_checks, result.success, scene.target_quat_wxyz)
                category, category_cn = failure_category(metrics.success, result.message, "planner")
                final_orientation_error = (
                    float(np.linalg.norm(orientation_error_vector(world.forward_quat(processed_path[-1]), scene.target_quat_wxyz)))
                    if processed_path and scene.target_quat_wxyz is not None
                    else 0.0
                )
                store.save_planner_result({
                    "trial_id": trial_id,
                    "global_seed": cfg.global_seed,
                    "scene_id": scene.id,
                    "scene_seed": scene.seed,
                    "planner": planner_name,
                    "trial_seed": seed,
                    "success": metrics.success,
                    "planning_time": metrics.planning_time,
                    "path_length_joint": metrics.path_length_joint,
                    "path_length_task": metrics.path_length_task,
                    "smoothness": metrics.smoothness,
                    "min_obstacle_distance": metrics.min_obstacle_distance,
                    "min_ground_clearance": metrics.metadata.get("min_ground_clearance", float("nan")),
                    "collision_checks": metrics.collision_checks,
                    "num_waypoints": metrics.num_waypoints,
                    "final_error": metrics.final_error,
                    "final_orientation_error": final_orientation_error,
                    "q_start": array_to_string(scene.q_start),
                    "q_goal": array_to_string(q_goal),
                    "target_position": array_to_string(scene.target_position),
                    "target_quat_wxyz": array_to_string(scene.target_quat_wxyz) if scene.target_quat_wxyz is not None else "",
                    "message": result.message,
                    "failure_category": category,
                    "failure_category_cn": category_cn,
                })
                if not metrics.success:
                    snapshot = _failure_snapshot_row(scene, planner_name, trial_id, seed, category, category_cn, result.message, result, processed_path, world)
                    if snapshot is not None:
                        store.save_planner_failure_snapshot(snapshot)
                if processed_path:
                    ee_points = world.sample_end_effector_path(processed_path)
                    store.save_trajectory(trajectory_rows(scene.id, planner_name, trial_id, processed_path, ee_points, world, scene.target_position, scene.target_quat_wxyz))
                    kin = trajectory_kinematic_metrics(processed_path, world, scene.target_position, scene.target_quat_wxyz)
                    store.save_trajectory_metrics({
                        "trial_id": trial_id,
                        "global_seed": cfg.global_seed,
                        "scene_id": scene.id,
                        "scene_seed": scene.seed,
                        "planner": planner_name,
                        "trial_seed": seed,
                        "success": metrics.success,
                        **kin,
                    })
                    if log_rerun and trial_id == 0:
                        logger.log_path(planner_name, ee_points)
                        logger.log_joint_series(planner_name, processed_path)
                        logger.log_metrics(planner_name, metrics)
                store.save_reproducibility_row({
                    "trial_id": trial_id,
                    "global_seed": cfg.global_seed,
                    "scene_id": scene.id,
                    "scene_seed": scene.seed,
                    "algorithm_type": "planner",
                    "algorithm": planner_name,
                    "trial_seed": seed,
                    "q_start": array_to_string(scene.q_start),
                    "q_goal": array_to_string(q_goal),
                    "target_position": array_to_string(scene.target_position),
                    "target_quat_wxyz": array_to_string(scene.target_quat_wxyz) if scene.target_quat_wxyz is not None else "",
                    "success": result.success,
                })


def run_experiments(
    trials: int | None = None,
    reset: bool = True,
    log_rerun: bool = True,
    config_path: str | None = None,
    scene_id: str | None = None,
) -> None:
    store = ResultStore()
    if reset:
        store.reset()
    validate_scenes(config_path=config_path, scene_id=scene_id)
    run_ik_suite(store, trials=trials, config_path=config_path, scene_id=scene_id)
    run_planner_suite(store, trials=trials, log_rerun=log_rerun, config_path=config_path, scene_id=scene_id)
    write_summary_tables(store)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _quantile(series: pd.Series, q: float) -> float:
    clean = _numeric(series).dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(q))


def _ci95_binary(series: pd.Series) -> float:
    clean = _numeric(series).dropna()
    n = len(clean)
    if n == 0:
        return float("nan")
    p = float(clean.mean())
    return float(1.96 * np.sqrt(p * (1.0 - p) / n))


def _message_mode(series: pd.Series) -> str:
    clean = series.dropna().astype(str)
    if clean.empty:
        return ""
    return str(clean.value_counts().index[0])


def _score_high(values: pd.Series) -> pd.Series:
    vals = _numeric(values)
    lo = vals.min(skipna=True)
    hi = vals.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi):
        return pd.Series(np.nan, index=values.index)
    if abs(float(hi - lo)) <= 1e-12:
        return pd.Series(1.0, index=values.index)
    return (vals - lo) / (hi - lo)


def _score_low(values: pd.Series) -> pd.Series:
    vals = _numeric(values)
    lo = vals.min(skipna=True)
    hi = vals.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi):
        return pd.Series(np.nan, index=values.index)
    if abs(float(hi - lo)) <= 1e-12:
        return pd.Series(1.0, index=values.index)
    return 1.0 - (vals - lo) / (hi - lo)


def _add_planner_scores(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    scored = summary.copy()
    parts = []
    for _, grp in scored.groupby("scene_id", sort=False):
        local = grp.copy()
        components = pd.DataFrame(index=local.index)
        components["success"] = _score_high(local["success_rate"])
        components["time"] = _score_low(local["planning_time_mean"])
        components["joint_length"] = _score_low(local["path_length_joint_mean"])
        components["smoothness"] = _score_low(local["smoothness_mean"])
        components["safety"] = _score_high(local["min_obstacle_distance_mean"])
        components["collision_checks"] = _score_low(local["collision_checks_mean"])
        components["final_error"] = _score_low(local["final_error_mean"])
        local["composite_score"] = components.mean(axis=1, skipna=True)
        local["composite_rank"] = local["composite_score"].rank(ascending=False, method="min")
        parts.append(local)
    return pd.concat(parts, ignore_index=True)


def write_summary_tables(store: ResultStore | None = None) -> None:
    store = store or ResultStore()
    if store.ik_path.exists():
        ik = pd.read_csv(store.ik_path)
        if not ik.empty:
            ik = ik.replace([np.inf, -np.inf], np.nan)
            ik_summary = ik.groupby(["scene_id", "ik_method"], as_index=False).agg(
                success_rate=("success", "mean"),
                success_ci95=("success", _ci95_binary),
                error_mean=("position_error", "mean"),
                error_std=("position_error", "std"),
                error_p50=("position_error", lambda s: _quantile(s, 0.50)),
                error_p90=("position_error", lambda s: _quantile(s, 0.90)),
                solve_time_mean=("solve_time", "mean"),
                solve_time_std=("solve_time", "std"),
                solve_time_p50=("solve_time", lambda s: _quantile(s, 0.50)),
                solve_time_p90=("solve_time", lambda s: _quantile(s, 0.90)),
                iterations_mean=("iterations", "mean"),
                condition_number_mean=("condition_number", "mean"),
                condition_number_p50=("condition_number", lambda s: _quantile(s, 0.50)),
                condition_number_p90=("condition_number", lambda s: _quantile(s, 0.90)),
                condition_number_max=("condition_number", "max"),
            )
            ik_summary.to_csv(store.ik_summary_path, index=False)
    if store.planner_path.exists():
        planner = pd.read_csv(store.planner_path)
        if not planner.empty:
            planner = planner.replace([np.inf, -np.inf], np.nan)
            for column in [
                "planning_time",
                "path_length_joint",
                "path_length_task",
                "smoothness",
                "min_obstacle_distance",
                "min_ground_clearance",
                "collision_checks",
                "final_error",
                "final_orientation_error",
                "num_waypoints",
            ]:
                if column not in planner.columns:
                    planner[column] = np.nan
            if "message" not in planner.columns:
                planner["message"] = ""
            if "failure_category" not in planner.columns or "failure_category_cn" not in planner.columns:
                categories = [failure_category(success, message, "planner") for success, message in zip(planner["success"], planner["message"].fillna(""))]
                planner["failure_category"] = [item[0] for item in categories]
                planner["failure_category_cn"] = [item[1] for item in categories]
            planner_summary = planner.groupby(["scene_id", "planner"], as_index=False).agg(
                success_rate=("success", "mean"),
                success_ci95=("success", _ci95_binary),
                failure_rate=("success", lambda s: 1.0 - float(_numeric(s).mean())),
                planning_time_mean=("planning_time", "mean"),
                planning_time_std=("planning_time", "std"),
                planning_time_p50=("planning_time", lambda s: _quantile(s, 0.50)),
                planning_time_p90=("planning_time", lambda s: _quantile(s, 0.90)),
                path_length_joint_mean=("path_length_joint", "mean"),
                path_length_joint_p50=("path_length_joint", lambda s: _quantile(s, 0.50)),
                path_length_joint_p90=("path_length_joint", lambda s: _quantile(s, 0.90)),
                path_length_task_mean=("path_length_task", "mean"),
                path_length_task_p50=("path_length_task", lambda s: _quantile(s, 0.50)),
                path_length_task_p90=("path_length_task", lambda s: _quantile(s, 0.90)),
                smoothness_mean=("smoothness", "mean"),
                smoothness_p50=("smoothness", lambda s: _quantile(s, 0.50)),
                smoothness_p90=("smoothness", lambda s: _quantile(s, 0.90)),
                min_obstacle_distance_mean=("min_obstacle_distance", "mean"),
                min_obstacle_distance_p10=("min_obstacle_distance", lambda s: _quantile(s, 0.10)),
                min_obstacle_distance_p50=("min_obstacle_distance", lambda s: _quantile(s, 0.50)),
                min_ground_clearance_mean=("min_ground_clearance", "mean"),
                min_ground_clearance_p10=("min_ground_clearance", lambda s: _quantile(s, 0.10)),
                collision_checks_mean=("collision_checks", "mean"),
                collision_checks_p90=("collision_checks", lambda s: _quantile(s, 0.90)),
                final_error_mean=("final_error", "mean"),
                final_error_p90=("final_error", lambda s: _quantile(s, 0.90)),
                final_orientation_error_mean=("final_orientation_error", "mean"),
                final_orientation_error_p90=("final_orientation_error", lambda s: _quantile(s, 0.90)),
                num_waypoints_mean=("num_waypoints", "mean"),
                message=("message", _message_mode),
                failure_category_cn=("failure_category_cn", _message_mode),
            )
            planner_summary["safety_margin_mean"] = planner_summary["min_obstacle_distance_mean"]
            planner_summary = _add_planner_scores(planner_summary)
            planner_summary.to_csv(store.planner_summary_path, index=False)
            failure_summary = (
                planner.assign(message=planner["message"].fillna(""))
                .groupby(["scene_id", "planner", "failure_category", "failure_category_cn", "message"], as_index=False)
                .agg(count=("success", "size"), success_rate=("success", "mean"))
            )
            totals = failure_summary.groupby(["scene_id", "planner"])["count"].transform("sum")
            failure_summary["share"] = failure_summary["count"] / totals.replace(0, np.nan)
            failure_summary.to_csv(store.planner_failure_summary_path, index=False)
            ranking = planner_summary.sort_values(["scene_id", "composite_rank", "planner"])
            ranking.to_csv(store.planner_ranking_path, index=False)


def run_demo(
    no_viewer: bool = False,
    config_path: str | None = None,
    ik_method: str | None = None,
    planner_name: str | None = None,
    scene_id: str | None = None,
) -> None:
    cfg = load_project_config(config_path=config_path)
    scene_id = scene_id or cfg.experiment.get("demo_scene", cfg.scenes[0].id)
    scene = next((s for s in cfg.scenes if s.id == scene_id), None)
    if scene is None:
        known = ", ".join(s.id for s in cfg.scenes)
        raise ValueError(f"Unknown scene: {scene_id}. Available scenes: {known}")
    ik_method = ik_method or cfg.ik.get("demo_method")
    planner_name = planner_name or cfg.planners.get("demo_method", "rrt_connect")
    scene, world, q_goal, result, attempts = demo_plan(scene, cfg, ik_method, planner_name)
    path = process_path(result.path, world, cfg.trajectory.get("shortcut_iterations", 80), cfg.trajectory.get("samples", 80), create_rng(cfg.global_seed + 999))
    if not _path_is_valid(path, world):
        raise RuntimeError("Demo postprocess invalid: collision or ground penetration")
    ee_points = world.sample_end_effector_path(path)
    metrics = compute_trial_metrics(path, world, scene.target_position, result.planning_time, result.collision_checks, result.success, scene.target_quat_wxyz)
    final_orientation_error = (
        float(np.linalg.norm(orientation_error_vector(world.forward_quat(path[-1]), scene.target_quat_wxyz)))
        if path and scene.target_quat_wxyz is not None
        else 0.0
    )
    joint_motion = float(np.linalg.norm(path[-1] - path[0])) if len(path) >= 2 else 0.0
    print(
        f"Demo success: scene={scene.id}, ik={ik_method}, planner={planner_name}, obstacles={len(scene.obstacles)}, attempts={attempts}, "
        f"waypoints={len(path)}, joint_motion={joint_motion:.3f} rad, "
        f"final_ee_error={metrics.final_error:.3f} m, final_orientation_error={final_orientation_error:.3f} rad, "
        f"planning_time={result.planning_time:.3f}s"
    )
    if not no_viewer:
        print("MuJoCo viewer is looping the trajectory. Close the viewer window to exit.")
    logger = RerunLogger(enabled=cfg.rerun.get("enabled", True), recording_path=cfg.rerun.get("recording_path", "results/recordings/arm_planning.rrd"), spawn=cfg.rerun.get("spawn", False))
    logger.log_scene(scene)
    demo_name = f"{planner_name}_demo"
    logger.log_path(demo_name, ee_points)
    logger.log_joint_series(demo_name, path)
    logger.log_metrics(demo_name, metrics)
    if not no_viewer:
        world.play_trajectory(path, ee_points=ee_points)


def _scene_by_id(cfg, scene_id: str):
    scene = next((s for s in cfg.scenes if s.id == scene_id), None)
    if scene is None:
        known = ", ".join(s.id for s in cfg.scenes)
        raise ValueError(f"Unknown scene: {scene_id}. Available scenes: {known}")
    return scene


def run_all_combos(
    config_path: str | None = None,
    scene_id: str | None = None,
    no_viewer: bool = False,
) -> list[dict]:
    cfg = load_project_config(config_path=config_path)
    selected_scene_id = scene_id or cfg.experiment.get("demo_scene", cfg.scenes[0].id)
    scene = _scene_by_id(cfg, selected_scene_id)
    ik_methods = list(cfg.ik.get("methods", ["scipy_baseline"]))
    planner_methods = list(cfg.planners.get("methods", ["rrt_connect"]))
    rows: list[dict] = []

    print(f"Running all combinations on scene={scene.id}: {len(ik_methods)} IK x {len(planner_methods)} planners")
    for ik_idx, ik_name in enumerate(ik_methods):
        for planner_idx, planner_name in enumerate(planner_methods):
            combo = f"IK={ik_name} planner={planner_name}"
            world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
            row = {
                "scene_id": scene.id,
                "ik_method": ik_name,
                "planner": planner_name,
                "success": False,
                "phase": "",
                "message": "",
                "final_error": float("nan"),
                "orientation_error": float("nan"),
                "min_obstacle_distance": float("nan"),
                "planning_time": float("nan"),
                "waypoints": 0,
                "playback_success": False,
            }
            ik = create_ik_solver(ik_name, cfg.ik).solve(scene.target_position, scene.q_start, world, scene.target_quat_wxyz)
            seed = cfg.global_seed + 1000 * (ik_idx + 1) + 37 * (planner_idx + 1)
            validation_message = "" if ik.success else ik.message
            if ik.success:
                validation_message = _validation_message(world, scene.q_start, ik.q)
            ik_valid = bool(ik.success and not validation_message)
            result = None
            if ik_valid:
                planner = create_planner(planner_name, cfg.planners)
                result = planner.plan(scene.q_start, ik.q, world, create_rng(seed))
            path = []
            if result is not None and result.success:
                path = process_path(
                    result.path,
                    world,
                    cfg.trajectory.get("shortcut_iterations", 80),
                    cfg.trajectory.get("samples", 80),
                    create_rng(seed + 999),
                )
                if not _path_is_valid(path, world):
                    result.success = False
                    result.message = "postprocess invalid: collision or ground penetration"
            elif result is not None and result.path:
                path = result.path

            if path and result is not None and result.success:
                metrics = compute_trial_metrics(
                    path,
                    world,
                    scene.target_position,
                    result.planning_time,
                    result.collision_checks,
                    result.success,
                    scene.target_quat_wxyz,
                )
                final_orientation_error = (
                    float(np.linalg.norm(orientation_error_vector(world.forward_quat(path[-1]), scene.target_quat_wxyz)))
                    if scene.target_quat_wxyz is not None
                    else 0.0
                )
                row.update(
                    {
                        "final_error": metrics.final_error,
                        "orientation_error": final_orientation_error,
                        "min_obstacle_distance": metrics.min_obstacle_distance,
                        "planning_time": result.planning_time,
                        "waypoints": metrics.num_waypoints,
                        "playback_success": True,
                    }
                )
            else:
                row.update({"planning_time": result.planning_time if result is not None else float("nan")})
            if not ik.success:
                phase = "ik"
                message = ik.message
            elif not ik_valid:
                phase = "validation"
                message = validation_message
            else:
                phase = "planner"
                message = result.message if result is not None else ""
            category, category_cn = failure_category(bool(result.success) if result is not None else False, message, phase)
            row.update({
                "success": bool(result.success) if result is not None else False,
                "phase": phase,
                "message": message,
                "failure_category": category,
                "failure_category_cn": category_cn,
            })
            rows.append(row)

            if path and row["playback_success"]:
                ee_points = world.sample_end_effector_path(path)
                print(
                    f"[ OK ] {combo}: waypoints={len(path)}, final_error={row['final_error']:.4f} m, "
                    f"orientation_error={row['orientation_error']:.4f} rad, "
                    f"min_distance={row['min_obstacle_distance']:.4f} m, planning_time={row['planning_time']:.3f}s"
                )
                if not no_viewer:
                    print("Playing this trajectory once. Close the viewer window to continue to the next combination.")
                    world.play_trajectory(path, ee_points=ee_points, loop=False, wait_after_playback=True)
            else:
                print(f"[FAIL] {combo}: category={row['failure_category_cn']}, message={row['message']}, planning_time={row['planning_time']:.3f}s")

    success_count = sum(1 for row in rows if row["success"])
    playback_count = sum(1 for row in rows if row["playback_success"])
    print(f"All-combo run complete: {success_count}/{len(rows)} planners succeeded, {playback_count}/{len(rows)} combinations have playback trajectories.")
    return rows
