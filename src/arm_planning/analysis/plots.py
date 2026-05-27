from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arm_planning.utils.config import resolve_path


plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK JP", "Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans"],
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
})

METHOD_COLORS = {
    "pinv": "#4C78A8",
    "dls": "#59A14F",
    "lm": "#F28E2B",
    "scipy_baseline": "#B07AA1",
    "rrt": "#4C78A8",
    "rrt_connect": "#59A14F",
    "prm": "#F28E2B",
    "apf": "#E15759",
}

SHORT_LABELS = {
    "scipy_baseline": "SciPy",
    "rrt_connect": "RRT-C",
    "roadmap_disconnected": "图未连通",
    "采样图未连通": "图未连通",
    "iteration_limit": "迭代上限",
    "local_minimum": "局部极小",
    "ground_penetration": "穿地",
    "obstacle_collision": "碰撞",
    "postprocess_invalid": "后处理无效",
}


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png")
    fig.savefig(output_dir / f"{name}.svg")
    plt.close(fig)


def _color_for(label: str) -> str:
    return METHOD_COLORS.get(str(label), "#6B7280")


def _short_label(label: object) -> str:
    text = str(label)
    return SHORT_LABELS.get(text, text)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in df.columns for column in columns)


def _boxplot(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, output_dir: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = list(df[x].dropna().unique())
    data = []
    for label in labels:
        values = pd.to_numeric(df.loc[df[x] == label, y], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        data.append(values if len(values) else np.asarray([np.nan]))
    ax.boxplot(data, tick_labels=[_short_label(label) for label in labels], showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_dir, name)


def _heatmap(table: pd.DataFrame, title: str, cbar_label: str, output_dir: Path, name: str, fmt: str = ".2f") -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    values = table.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(table.columns)), labels=[_short_label(x) for x in table.columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(table.index)), labels=[_short_label(x) for x in table.index])
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            label = "nan" if np.isnan(val) else format(val, fmt)
            ax.text(j, i, label, ha="center", va="center", color="white" if not np.isnan(val) and val < np.nanmax(values) * 0.65 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    _save(fig, output_dir, name)


def _safe_pivot(df: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame:
    clean = df.copy()
    clean[values] = pd.to_numeric(clean[values], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return clean.pivot_table(index=index, columns=columns, values=values, aggfunc="mean").sort_index()


def _summary_table_image(df: pd.DataFrame, output_dir: Path, name: str, title: str, columns: list[str]) -> None:
    if df.empty:
        return
    shown = df[columns].copy()
    for column in shown.columns:
        if column in {"planner", "ik_method", "failure_category", "failure_category_cn", "message"}:
            shown[column] = shown[column].map(_short_label)
    for column in shown.columns:
        if shown[column].dtype.kind in "fc":
            shown[column] = shown[column].map(lambda x: "" if pd.isna(x) else f"{x:.3g}")
    fig, ax = plt.subplots(figsize=(min(11.5, 1.3 * len(columns) + 2.0), min(7.5, 0.38 * len(shown) + 1.6)))
    ax.axis("off")
    ax.set_title(title, loc="left", pad=10)
    table = ax.table(
        cellText=shown.to_numpy(),
        colLabels=shown.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    _save(fig, output_dir, name)


def export_mujoco_scene_screenshots(output_dir: str | Path = "images/generated", config_path: str | Path | None = None) -> None:
    """Render deterministic offscreen screenshots of the configured MuJoCo scenes."""
    try:
        import mujoco
    except Exception as exc:
        print(f"Skipping MuJoCo screenshots: {exc}")
        return

    from arm_planning.sim.mujoco_world import world_for_scene
    from arm_planning.utils.config import load_project_config

    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_project_config(config_path=config_path)
    for scene in cfg.scenes:
        try:
            world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
            world.set_qpos(scene.q_start)
            renderer = mujoco.Renderer(world.model, height=480, width=640)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = np.array([0.36, 0.0, 0.30])
            camera.distance = 1.65
            camera.azimuth = 135
            camera.elevation = -25
            renderer.update_scene(world.data, camera=camera)
            image = renderer.render()
            renderer.close()
            plt.imsave(output / f"mujoco_scene_{scene.id}.png", image)
            fig, ax = plt.subplots(figsize=(6.8, 4.6))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(scene.id)
            _save(fig, output, f"mujoco_scene_{scene.id}")
        except Exception as exc:
            print(f"Skipping MuJoCo screenshot for {scene.id}: {exc}")


def _failure_snapshot_candidates(df: pd.DataFrame, limit: int = 4) -> pd.DataFrame:
    priority = {
        "local_minimum": 0,
        "iteration_limit": 1,
        "roadmap_disconnected": 2,
        "obstacle_collision": 3,
        "ground_penetration": 4,
        "postprocess_invalid": 5,
        "invalid_start_or_goal": 6,
        "ik_not_converged": 7,
        "other": 8,
    }
    clean = df.copy()
    clean["_priority"] = clean.get("failure_category", "").map(priority).fillna(99)
    clean["_ee_error"] = pd.to_numeric(clean.get("ee_error", np.nan), errors="coerce")
    clean = clean.sort_values(["_priority", "_ee_error"], ascending=[True, False])
    rows = []
    seen: set[str] = set()
    for _, row in clean.iterrows():
        category = str(row.get("failure_category", "other"))
        if category in seen:
            continue
        rows.append(row)
        seen.add(category)
        if len(rows) >= limit:
            break
    if not rows and not clean.empty:
        rows = [clean.iloc[0]]
    return pd.DataFrame(rows).drop(columns=["_priority", "_ee_error"], errors="ignore")


def _render_mujoco_snapshot(world, q: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    import mujoco

    world.set_qpos(q)
    renderer = mujoco.Renderer(world.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.36, 0.0, 0.30])
    camera.distance = 1.55
    camera.azimuth = 135
    camera.elevation = -24
    renderer.update_scene(world.data, camera=camera)
    image = renderer.render()
    renderer.close()
    return image


def export_mujoco_failure_snapshots(
    failure_csv: str | Path = "results/data/planner_failure_snapshots.csv",
    output_dir: str | Path = "images/generated",
    config_path: str | Path | None = None,
) -> None:
    """Render representative MuJoCo states for failed planning trials."""
    try:
        import mujoco  # noqa: F401
    except Exception as exc:
        print(f"Skipping MuJoCo failure snapshots: {exc}")
        return

    path = resolve_path(failure_csv)
    output = resolve_path(output_dir)
    if not path.exists():
        print(f"Skipping MuJoCo failure snapshots: {path} not found")
        return
    df = pd.read_csv(path)
    if df.empty or not _require_columns(df, ["scene_id", "planner", "q_snapshot", "failure_category_cn"]):
        print("Skipping MuJoCo failure snapshots: no failed trial snapshots")
        return

    from arm_planning.sim.mujoco_world import world_for_scene
    from arm_planning.utils.config import load_project_config

    cfg = load_project_config(config_path=config_path)
    scenes = {scene.id: scene for scene in cfg.scenes}
    images: list[tuple[pd.Series, np.ndarray]] = []
    for _, row in _failure_snapshot_candidates(df, limit=4).iterrows():
        scene = scenes.get(str(row["scene_id"]))
        if scene is None:
            continue
        try:
            world = world_for_scene(cfg.robot, scene, cfg.experiment.get("edge_resolution", 0.06))
            q = _parse_q(row["q_snapshot"])
            images.append((row, _render_mujoco_snapshot(world, q)))
        except Exception as exc:
            print(f"Skipping failure snapshot for {row.get('scene_id')} / {row.get('planner')}: {exc}")

    if not images:
        print("Skipping MuJoCo failure snapshots: no renderable failed trial snapshots")
        return

    cols = min(2, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.8 * cols, 5.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for idx, (row, image) in enumerate(images):
        ax = axes[idx // cols][idx % cols]
        ax.imshow(image)
        title = f"{_short_label(row['planner'])} | {_short_label(row['failure_category_cn'])}"
        detail = f"ee error={float(row.get('ee_error', np.nan)):.3f} m, min dist={float(row.get('min_obstacle_distance', np.nan)):.3f} m"
        ax.set_title(f"{title}\n{detail}", loc="left", fontsize=10)
        ax.axis("off")
    _save(fig, output, "mujoco_failure_snapshots")

    for idx, (row, image) in enumerate(images, start=1):
        fig, ax = plt.subplots(figsize=(8.4, 5.8))
        ax.imshow(image)
        ax.axis("off")
        title = f"Failure snapshot {idx}: {_short_label(row['planner'])} / {_short_label(row['failure_category_cn'])}"
        detail = f"scene={row['scene_id']}, error={float(row.get('ee_error', np.nan)):.3f} m, min distance={float(row.get('min_obstacle_distance', np.nan)):.3f} m"
        ax.set_title(f"{title}\n{detail}", loc="left", fontsize=11)
        _save(fig, output, f"mujoco_failure_snapshot_{idx:02d}")


def export_ik_charts(ik_csv: str | Path = "results/data/ik_results.csv", output_dir: str | Path = "images/generated") -> None:
    path = resolve_path(ik_csv)
    output = resolve_path(output_dir)
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    df = df.replace([np.inf, -np.inf], np.nan)
    summary = df.groupby("ik_method", as_index=False).agg(
        position_error=("position_error", "mean"),
        solve_time=("solve_time", "mean"),
        iterations=("iterations", "mean"),
        condition_number=("condition_number", "mean"),
        success=("success", "mean"),
    )
    for metric, ylabel, name, title in [
        ("position_error", "Mean position error / m", "ik_error_compare", "IK position error comparison"),
        ("solve_time", "Mean solve time / s", "ik_time_compare", "IK solve time comparison"),
        ("iterations", "Mean iterations", "ik_iterations_compare", "IK iteration comparison"),
        ("condition_number", "Mean Jacobian condition number", "ik_condition_compare", "IK condition number comparison"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar(summary["ik_method"], summary[metric], color="#4C78A8")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        _save(fig, output, name)
    _boxplot(df, "ik_method", "position_error", "IK position error distribution", "Position error / m", output, "ik_error_boxplot")
    _boxplot(df, "ik_method", "solve_time", "IK solve time distribution", "Solve time / s", output, "ik_time_boxplot")
    _boxplot(df, "ik_method", "iterations", "IK iteration distribution", "Iterations", output, "ik_iterations_boxplot")
    _heatmap(_safe_pivot(df, "scene_id", "ik_method", "success"), "IK success rate by scene", "Success rate", output, "ik_success_heatmap")
    _heatmap(_safe_pivot(df, "scene_id", "ik_method", "position_error"), "IK error by scene", "Position error / m", output, "ik_error_heatmap", fmt=".3g")


def export_planner_charts(planner_csv: str | Path = "results/data/planner_results.csv", output_dir: str | Path = "images/generated") -> None:
    path = resolve_path(planner_csv)
    output = resolve_path(output_dir)
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    df = df.replace([np.inf, -np.inf], np.nan)
    summary = df.groupby("planner", as_index=False).agg(
        success_rate=("success", "mean"),
        planning_time=("planning_time", "mean"),
        path_length_joint=("path_length_joint", "mean"),
        smoothness=("smoothness", "mean"),
        min_obstacle_distance=("min_obstacle_distance", "mean"),
        collision_checks=("collision_checks", "mean"),
    )
    charts = [
        ("success_rate", "Success rate", "planner_success_rate", "Planner success rate comparison"),
        ("path_length_joint", "Mean joint-space path length / rad", "path_length_compare", "Joint-space path length comparison"),
        ("smoothness", "Mean second-difference smoothness", "smoothness_compare", "Trajectory smoothness comparison"),
        ("min_obstacle_distance", "Mean minimum obstacle distance / m", "min_distance_compare", "Minimum obstacle distance comparison"),
        ("collision_checks", "Mean collision checks", "collision_checks_compare", "Collision-check count comparison"),
    ]
    for metric, ylabel, name, title in charts:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar([_short_label(x) for x in summary["planner"]], summary[metric], color="#59A14F")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        _save(fig, output, name)
    for metric, ylabel, name in [
        ("planning_time", "Planning time / s", "planner_time_boxplot"),
        ("path_length_joint", "Joint path length / rad", "planner_path_length_boxplot"),
        ("smoothness", "Second-difference smoothness", "planner_smoothness_boxplot"),
        ("min_obstacle_distance", "Minimum obstacle distance / m", "planner_min_distance_boxplot"),
        ("collision_checks", "Collision checks", "planner_collision_checks_boxplot"),
    ]:
        _boxplot(df, "planner", metric, f"Planner {metric} distribution", ylabel, output, name)
    _heatmap(_safe_pivot(df, "scene_id", "planner", "success"), "Planner success rate by scene", "Success rate", output, "planner_success_heatmap")
    _heatmap(_safe_pivot(df, "scene_id", "planner", "planning_time"), "Planner time by scene", "Planning time / s", output, "planner_time_heatmap")

    radar_metrics = ["success_rate", "planning_time", "path_length_joint", "smoothness", "collision_checks"]
    normalized = summary.set_index("planner")[radar_metrics].copy()
    for col in radar_metrics:
        vals = normalized[col].astype(float)
        if col == "success_rate":
            normalized[col] = vals / max(vals.max(), 1e-12)
        else:
            normalized[col] = 1.0 - (vals - vals.min()) / max(vals.max() - vals.min(), 1e-12)
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw={"polar": True})
    for planner, row in normalized.iterrows():
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, linewidth=1.8, label=_short_label(planner))
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1], radar_metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Normalized planner profile")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.08))
    _save(fig, output, "planner_radar_profile")


def _parse_q(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    return np.asarray([float(x) for x in str(value).strip("[]").split(",") if x.strip()], dtype=float)


def _yaw_from_quat_wxyz(quat: np.ndarray | None) -> float:
    if quat is None:
        return 0.0
    q = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return 0.0
    w, x, y, z = q / norm
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _add_obstacle_top_view(ax, obs, alpha: float = 0.28) -> None:
    if obs.type == "box" and obs.size is not None:
        half = np.asarray(obs.size[:2], dtype=float) / 2.0
        lower_left = np.asarray(obs.position[:2], dtype=float) - half
        rect = plt.Rectangle(
            lower_left,
            2 * half[0],
            2 * half[1],
            angle=np.degrees(_yaw_from_quat_wxyz(obs.quat_wxyz)),
            rotation_point=tuple(obs.position[:2]),
            color="#D55E00",
            alpha=alpha,
        )
        ax.add_patch(rect)
    elif obs.type == "sphere" and obs.radius is not None:
        circle = plt.Circle(obs.position[:2], obs.radius, color="#D55E00", alpha=alpha)
        ax.add_patch(circle)


def _plot_obstacle_markers_3d(ax, obstacles) -> None:
    for obs in obstacles:
        if obs.type == "box" and obs.size is not None:
            ax.scatter(obs.position[0], obs.position[1], obs.position[2], color="#D55E00", s=18, alpha=0.78)
        elif obs.type == "sphere" and obs.radius is not None:
            ax.scatter(obs.position[0], obs.position[1], obs.position[2], color="#D55E00", s=34, alpha=0.82)


def _obstacles_for_scene(scene_id: str, config_path: str | Path | None = None):
    from arm_planning.utils.config import load_project_config

    cfg = load_project_config(config_path=config_path)
    scene = next((s for s in cfg.scenes if s.id == scene_id), None)
    return [] if scene is None else scene.obstacles, None if scene is None else scene.target_position


def _first_successful_trajectory(df: pd.DataFrame, preferred_scene: str = "scene_pass_through_maze", preferred_planner: str = "rrt_connect") -> pd.DataFrame:
    candidates = df.copy()
    if preferred_scene in set(candidates["scene_id"]):
        candidates = candidates[candidates["scene_id"] == preferred_scene]
    if preferred_planner in set(candidates["planner"]):
        candidates = candidates[candidates["planner"] == preferred_planner]
    for _, group in candidates.groupby(["scene_id", "planner", "trial_id"], sort=False):
        ordered = group.sort_values("sample_index")
        if len(ordered) >= 2:
            return ordered
    return pd.DataFrame()


def export_paper_trajectory_figures(
    traj_csv: str | Path = "results/data/trajectory_samples.csv",
    output_dir: str | Path = "images/generated",
    config_path: str | Path | None = None,
) -> None:
    path = resolve_path(traj_csv)
    output = resolve_path(output_dir)
    if not path.exists():
        return
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    required = ["scene_id", "planner", "trial_id", "sample_index", "q", "ee_x", "ee_y", "ee_z"]
    if df.empty or not _require_columns(df, required):
        return
    traj = _first_successful_trajectory(df)
    if traj.empty:
        return
    scene_id = str(traj["scene_id"].iloc[0])
    planner = str(traj["planner"].iloc[0])
    obstacles, target = _obstacles_for_scene(scene_id, config_path=config_path)
    ee = traj[["ee_x", "ee_y", "ee_z"]].to_numpy(float)
    sample = traj["sample_index"].to_numpy(int)
    color = _color_for(planner)

    fig = plt.figure(figsize=(10.5, 4.8))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=color, linewidth=2.0, label=f"{_short_label(planner)} path")
    ax3d.scatter(ee[0, 0], ee[0, 1], ee[0, 2], color="#4C78A8", s=35, label="start")
    ax3d.scatter(ee[-1, 0], ee[-1, 1], ee[-1, 2], color="#59A14F", s=35, label="end")
    if target is not None:
        ax3d.scatter(target[0], target[1], target[2], color="#2CA02C", marker="*", s=80, label="target")
    _plot_obstacle_markers_3d(ax3d, obstacles)
    ax3d.set_title("End-effector path through mixed maze")
    ax3d.set_xlabel("x / m")
    ax3d.set_ylabel("y / m")
    ax3d.set_zlabel("z / m")
    ax3d.legend(fontsize=8)

    ax = fig.add_subplot(1, 2, 2)
    for obs in obstacles:
        _add_obstacle_top_view(ax, obs)
    ax.plot(ee[:, 0], ee[:, 1], color=color, linewidth=2.2)
    ax.scatter([ee[0, 0], ee[-1, 0]], [ee[0, 1], ee[-1, 1]], c=["#4C78A8", "#59A14F"], s=35)
    if target is not None:
        ax.scatter(target[0], target[1], marker="*", color="#2CA02C", s=90)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Top view: obstacle corridor")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(alpha=0.22)
    _save(fig, output, "paper_mixed_maze_ee_path")

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 7.0), sharex=True)
    for idx, label in enumerate(["x", "y", "z"]):
        axes[idx].plot(sample, ee[:, idx], color=color, linewidth=1.8)
        axes[idx].set_ylabel(f"EE {label} / m")
        axes[idx].grid(alpha=0.22)
    axes[0].set_title("End-effector Cartesian trajectory")
    axes[-1].set_xlabel("Sample index")
    _save(fig, output, "paper_ee_position_components")

    q_rows = []
    for _, row in traj.iterrows():
        for joint, value in enumerate(_parse_q(row["q"]), start=1):
            q_rows.append({"sample_index": int(row["sample_index"]), "joint": f"q{joint}", "value": value})
    qdf = pd.DataFrame(q_rows)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for joint, group in qdf.groupby("joint"):
        ax.plot(group["sample_index"], group["value"], linewidth=1.5, label=joint)
    ax.set_title("Joint-angle trajectories")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Joint angle / rad")
    ax.grid(alpha=0.22)
    ax.legend(ncol=3)
    _save(fig, output, "paper_joint_angles")

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.6), sharex=True)
    if "joint_speed_norm" in traj:
        axes[0].plot(sample, pd.to_numeric(traj["joint_speed_norm"], errors="coerce"), color="#4C78A8", linewidth=1.8, label="speed")
    if "joint_acc_norm" in traj:
        axes[0].plot(sample, pd.to_numeric(traj["joint_acc_norm"], errors="coerce"), color="#F28E2B", linewidth=1.5, label="acceleration")
    axes[0].set_ylabel("Joint increment norm")
    axes[0].set_title("Joint-space derivatives")
    axes[0].grid(alpha=0.22)
    axes[0].legend()
    if "ee_error" in traj:
        axes[1].plot(sample, pd.to_numeric(traj["ee_error"], errors="coerce"), color="#E15759", linewidth=1.8, label="terminal error")
    if "ee_orientation_error" in traj:
        axes[1].plot(sample, pd.to_numeric(traj["ee_orientation_error"], errors="coerce"), color="#B07AA1", linewidth=1.5, label="orientation error")
    if "min_obstacle_distance" in traj:
        axes[1].plot(sample, pd.to_numeric(traj["min_obstacle_distance"], errors="coerce"), color="#59A14F", linewidth=1.5, label="obstacle distance")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Distance / m")
    axes[1].grid(alpha=0.22)
    axes[1].legend()
    _save(fig, output, "paper_error_and_safety_curves")

    fig = plt.figure(figsize=(8.0, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=color, linewidth=2.4, label=f"{_short_label(planner)} path")
    ax.scatter(ee[0, 0], ee[0, 1], ee[0, 2], color="#4C78A8", s=48, label="start")
    ax.scatter(ee[-1, 0], ee[-1, 1], ee[-1, 2], color="#59A14F", s=48, label="end")
    if target is not None:
        ax.scatter(target[0], target[1], target[2], color="#2CA02C", marker="*", s=110, label="target")
    _plot_obstacle_markers_3d(ax, obstacles)
    ax.set_title("End-effector 3D path")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_zlabel("z / m")
    ax.legend(fontsize=9)
    _save(fig, output, "paper_ee_path_3d_large")

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for obs in obstacles:
        _add_obstacle_top_view(ax, obs, alpha=0.30)
    ax.plot(ee[:, 0], ee[:, 1], color=color, linewidth=2.6)
    ax.scatter([ee[0, 0], ee[-1, 0]], [ee[0, 1], ee[-1, 1]], c=["#4C78A8", "#59A14F"], s=48)
    if target is not None:
        ax.scatter(target[0], target[1], marker="*", color="#2CA02C", s=110)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Top view of mixed obstacle passage")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(alpha=0.22)
    _save(fig, output, "paper_ee_path_top_view_large")

    for idx, label in enumerate(["x", "y", "z"]):
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.plot(sample, ee[:, idx], color=color, linewidth=2.2)
        ax.set_title(f"End-effector {label}-position")
        ax.set_xlabel("Sample index")
        ax.set_ylabel(f"EE {label} / m")
        ax.grid(alpha=0.22)
        _save(fig, output, f"paper_ee_position_{label}")

    for column, ylabel, name, line_color in [
        ("ee_error", "Position error / m", "paper_ee_position_error", "#E15759"),
        ("ee_orientation_error", "Orientation error / rad", "paper_ee_orientation_error", "#B07AA1"),
        ("cumulative_task_length", "Cumulative task length / m", "paper_cumulative_task_length", "#2F4B7C"),
        ("min_obstacle_distance", "Minimum obstacle distance / m", "paper_min_obstacle_distance", "#59A14F"),
        ("min_ground_clearance", "Minimum ground clearance / m", "paper_min_ground_clearance", "#8CD17D"),
        ("joint_speed_norm", "Joint speed norm", "paper_joint_speed_norm", "#4C78A8"),
        ("joint_acc_norm", "Joint acceleration norm", "paper_joint_acc_norm", "#F28E2B"),
        ("joint_jerk_norm", "Joint jerk norm", "paper_joint_jerk_norm", "#79706E"),
    ]:
        if column in traj:
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            ax.plot(sample, pd.to_numeric(traj[column], errors="coerce"), color=line_color, linewidth=2.2)
            ax.set_title(ylabel)
            ax.set_xlabel("Sample index")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.22)
            _save(fig, output, name)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for joint, group in qdf.groupby("joint"):
        ax.plot(group["sample_index"], group["value"], linewidth=1.8, label=joint)
    ax.set_title("Joint-angle trajectories")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Joint angle / rad")
    ax.grid(alpha=0.22)
    ax.legend(ncol=3)
    _save(fig, output, "paper_joint_angles_large")


def export_trajectory_charts(traj_csv: str | Path = "results/data/trajectory_samples.csv", output_dir: str | Path = "images/generated") -> None:
    path = resolve_path(traj_csv)
    output = resolve_path(output_dir)
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    df = df.replace([np.inf, -np.inf], np.nan)
    scenes = list(df["scene_id"].dropna().unique())
    for scene_id in scenes:
        subset = df[(df["scene_id"] == scene_id) & (df["trial_id"] == 0)].copy()
        if subset.empty:
            continue
        try:
            fig = plt.figure(figsize=(7.0, 5.2))
            ax = fig.add_subplot(111, projection="3d")
            for planner, grp in subset.groupby("planner"):
                ax.plot(grp["ee_x"], grp["ee_y"], grp["ee_z"], label=_short_label(planner), linewidth=1.8)
            ax.set_title(f"End-effector trajectory: {scene_id}")
            ax.set_xlabel("x / m")
            ax.set_ylabel("y / m")
            ax.set_zlabel("z / m")
            ax.legend(loc="best")
        except Exception:
            fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
            for planner, grp in subset.groupby("planner"):
                axes[0].plot(grp["ee_x"], grp["ee_y"], label=_short_label(planner), linewidth=1.8)
                axes[1].plot(grp["ee_x"], grp["ee_z"], label=_short_label(planner), linewidth=1.8)
            axes[0].set_title(f"{scene_id}: x-y projection")
            axes[0].set_xlabel("x / m")
            axes[0].set_ylabel("y / m")
            axes[1].set_title(f"{scene_id}: x-z projection")
            axes[1].set_xlabel("x / m")
            axes[1].set_ylabel("z / m")
            for ax in axes:
                ax.grid(alpha=0.25)
            axes[1].legend(loc="best")
        name = "ee_trajectory_3d" if scene_id == scenes[0] else f"ee_trajectory_{scene_id}"
        _save(fig, output, name)

    first_scene = scenes[0]
    subset = df[(df["scene_id"] == first_scene) & (df["trial_id"] == 0)].copy()
    q_rows = []
    for _, row in subset.iterrows():
        for j, value in enumerate(_parse_q(row["q"])):
            q_rows.append({"planner": row["planner"], "sample_index": row["sample_index"], "joint": f"q{j + 1}", "value": value})
    qdf = pd.DataFrame(q_rows)
    if not qdf.empty:
        chosen = "rrt_connect" if "rrt_connect" in qdf["planner"].unique() else qdf["planner"].iloc[0]
        chosen_df = qdf[qdf["planner"] == chosen]
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        for joint, grp in chosen_df.groupby("joint"):
            ax.plot(grp["sample_index"], grp["value"], label=joint)
        ax.set_title(f"{chosen} joint-angle trajectories")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Joint angle / rad")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3)
        _save(fig, output, "joint_angles_curve")

        vel_rows = []
        acc_rows = []
        for joint, grp in chosen_df.groupby("joint"):
            vals = grp.sort_values("sample_index")["value"].to_numpy(float)
            vel = np.diff(vals)
            acc = np.diff(vals, n=2)
            vel_rows.append((joint, vel))
            acc_rows.append((joint, acc))
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        for joint, vel in vel_rows:
            ax.plot(np.arange(len(vel)), vel, label=joint)
        ax.set_title(f"{chosen} joint velocity increments")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Delta q / sample")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3)
        _save(fig, output, "joint_velocity_curve")

        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        for joint, acc in acc_rows:
            ax.plot(np.arange(len(acc)), acc, label=joint)
        ax.set_title(f"{chosen} joint acceleration increments")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Delta2 q / sample")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3)
        _save(fig, output, "joint_acceleration_curve")


def export_trajectory_metric_charts(metrics_csv: str | Path = "results/data/trajectory_metrics.csv", output_dir: str | Path = "images/generated") -> None:
    path = resolve_path(metrics_csv)
    output = resolve_path(output_dir)
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    for metric, ylabel, name in [
        ("max_joint_speed_norm", "Max joint speed increment", "trajectory_max_speed_boxplot"),
        ("max_joint_acc_norm", "Max joint acceleration increment", "trajectory_max_acc_boxplot"),
        ("max_joint_jerk_norm", "Max joint jerk increment", "trajectory_max_jerk_boxplot"),
        ("mean_joint_jerk_norm", "Mean joint jerk increment", "trajectory_mean_jerk_boxplot"),
        ("min_obstacle_distance", "Minimum obstacle distance / m", "trajectory_min_distance_boxplot"),
    ]:
        if metric in df.columns:
            _boxplot(df, "planner", metric, f"Trajectory {metric} distribution", ylabel, output, name)

    scene = df["scene_id"].iloc[0]
    subset = df[(df["scene_id"] == scene) & (df["trial_id"] == 0)]
    if not subset.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar([_short_label(x) for x in subset["planner"]], subset["min_obstacle_distance"], color="#F28E2B")
        ax.set_title(f"Minimum distance along first-trial paths: {scene}")
        ax.set_ylabel("Minimum obstacle distance / m")
        ax.grid(axis="y", alpha=0.25)
        _save(fig, output, "safety_distance_curve")

    summary_metrics = ["path_length_task", "min_obstacle_distance", "final_error", "mean_joint_jerk_norm", "smoothness"]
    available = [col for col in summary_metrics if col in df.columns]
    if available and "planner" in df.columns:
        summary = df.groupby("planner", as_index=False)[available].mean(numeric_only=True)
        fig, axes = plt.subplots(1, len(available), figsize=(3.2 * len(available), 3.8))
        axes = np.atleast_1d(axes)
        labels = {
            "path_length_task": "Task path / m",
            "min_obstacle_distance": "Min distance / m",
            "final_error": "Final error / m",
            "mean_joint_jerk_norm": "Mean jerk",
        }
        for ax, metric in zip(axes, available):
            ax.bar([_short_label(x) for x in summary["planner"]], summary[metric], color=[_color_for(x) for x in summary["planner"]])
            ax.set_title(labels.get(metric, metric))
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.22)
        _save(fig, output, "paper_trajectory_metric_summary")

        for metric, title, ylabel, name in [
            ("path_length_task", "Task-space path length", "Path length / m", "paper_metric_path_length"),
            ("smoothness", "Trajectory smoothness", "Second-difference smoothness", "paper_metric_smoothness"),
            ("final_error", "Final position error", "Final error / m", "paper_metric_final_error"),
        ]:
            if metric in summary.columns:
                fig, ax = plt.subplots(figsize=(7.5, 5.0))
                ax.bar([_short_label(x) for x in summary["planner"]], summary[metric], color=[_color_for(x) for x in summary["planner"]])
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.grid(axis="y", alpha=0.22)
                ax.tick_params(axis="x", rotation=20)
                _save(fig, output, name)


def export_extended_ik_charts(ik_csv: str | Path = "results/data/ik_results.csv", output_dir: str | Path = "images/generated") -> None:
    path = resolve_path(ik_csv)
    output = resolve_path(output_dir)
    if not path.exists():
        return
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    if df.empty or not _require_columns(df, ["ik_method", "condition_number", "position_error", "scene_id"]):
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for method, grp in df.groupby("ik_method"):
        ax.scatter(
            pd.to_numeric(grp["condition_number"], errors="coerce"),
            pd.to_numeric(grp["position_error"], errors="coerce"),
            s=24,
            alpha=0.62,
            label=method,
            color=_color_for(method),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Jacobian condition number")
    ax.set_ylabel("Position error / m")
    ax.set_title("IK conditioning and terminal error")
    ax.grid(alpha=0.25, which="both")
    ax.legend(ncol=2)
    _save(fig, output, "ik_condition_error_scatter")

    condition_table = _safe_pivot(df, "scene_id", "ik_method", "condition_number")
    _heatmap(condition_table, "Mean Jacobian condition number by scene", "Condition number", output, "ik_condition_heatmap", fmt=".2g")


def export_extended_planner_charts(
    planner_csv: str | Path = "results/data/planner_results.csv",
    summary_csv: str | Path = "results/data/planner_summary.csv",
    failure_csv: str | Path = "results/data/planner_failure_summary.csv",
    ranking_csv: str | Path = "results/data/planner_ranking.csv",
    output_dir: str | Path = "images/generated",
) -> None:
    planner_path = resolve_path(planner_csv)
    summary_path = resolve_path(summary_csv)
    failure_path = resolve_path(failure_csv)
    ranking_path = resolve_path(ranking_csv)
    output = resolve_path(output_dir)
    if not planner_path.exists():
        return
    df = pd.read_csv(planner_path).replace([np.inf, -np.inf], np.nan)
    if df.empty:
        return

    if _require_columns(df, ["planner", "planning_time", "path_length_joint", "success"]):
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        for planner, grp in df.groupby("planner"):
            ax.scatter(
                pd.to_numeric(grp["planning_time"], errors="coerce"),
                pd.to_numeric(grp["path_length_joint"], errors="coerce"),
                c=_color_for(planner),
                s=28,
                alpha=0.58,
                label=_short_label(planner),
            )
        ax.set_xlabel("Planning time / s")
        ax.set_ylabel("Joint-space path length / rad")
        ax.set_title("Planning time versus path length")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        _save(fig, output, "planner_time_length_scatter")

    if _require_columns(df, ["planner", "min_obstacle_distance", "smoothness"]):
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        for planner, grp in df.groupby("planner"):
            ax.scatter(
                pd.to_numeric(grp["min_obstacle_distance"], errors="coerce"),
                pd.to_numeric(grp["smoothness"], errors="coerce"),
                c=_color_for(planner),
                s=28,
                alpha=0.58,
                label=_short_label(planner),
            )
        ax.set_xlabel("Minimum obstacle distance / m")
        ax.set_ylabel("Second-difference smoothness")
        ax.set_title("Safety margin versus path smoothness")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        _save(fig, output, "planner_safety_smoothness_scatter")

    if summary_path.exists():
        summary = pd.read_csv(summary_path).replace([np.inf, -np.inf], np.nan)
        if not summary.empty and "composite_score" in summary.columns:
            score_table = _safe_pivot(summary, "scene_id", "planner", "composite_score")
            _heatmap(score_table, "Normalized composite score by scene", "Composite score", output, "planner_composite_heatmap")
            mean_scores = summary.groupby("planner", as_index=False)["composite_score"].mean().sort_values("composite_score", ascending=False)
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            ax.bar([_short_label(x) for x in mean_scores["planner"]], mean_scores["composite_score"], color=[_color_for(x) for x in mean_scores["planner"]])
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Mean normalized score")
            ax.set_title("Planner composite ranking")
            ax.grid(axis="y", alpha=0.25)
            _save(fig, output, "planner_composite_ranking")
            _summary_table_image(
                summary.sort_values(["scene_id", "composite_rank", "planner"]),
                output,
                "planner_extended_summary_table",
                "Extended planner summary",
                [
                    "scene_id",
                    "planner",
                    "success_rate",
                    "success_ci95",
                    "planning_time_p50",
                    "path_length_joint_p50",
                    "min_obstacle_distance_p10",
                    "composite_score",
                ],
            )

    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path).replace([np.inf, -np.inf], np.nan)
        if not ranking.empty and "composite_score" in ranking.columns:
            top = ranking.groupby("planner", as_index=False).agg(
                composite_score=("composite_score", "mean"),
                success_rate=("success_rate", "mean"),
                planning_time_p50=("planning_time_p50", "mean"),
                min_obstacle_distance_p10=("min_obstacle_distance_p10", "mean"),
            )
            _summary_table_image(
                top.sort_values("composite_score", ascending=False),
                output,
                "planner_ranking_table",
                "Planner ranking overview",
                ["planner", "composite_score", "success_rate", "planning_time_p50", "min_obstacle_distance_p10"],
            )

    if failure_path.exists():
        failures = pd.read_csv(failure_path)
        category_column = "failure_category_cn" if "failure_category_cn" in failures.columns else "message"
        if not failures.empty and _require_columns(failures, ["planner", category_column, "count"]):
            pivot = failures.pivot_table(index="planner", columns=category_column, values="count", aggfunc="sum", fill_value=0)
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(8.4, 4.8))
                bottom = np.zeros(len(pivot.index))
                palette = plt.cm.Set2(np.linspace(0, 1, len(pivot.columns)))
                for color, message in zip(palette, pivot.columns):
                    values = pivot[message].to_numpy(float)
                    ax.bar([_short_label(x) for x in pivot.index], values, bottom=bottom, label=_short_label(message), color=color)
                    bottom += values
                ax.set_ylabel("Trial count")
                ax.set_title("Planner outcomes by message")
                ax.grid(axis="y", alpha=0.25)
                ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
                _save(fig, output, "planner_failure_stacked")


def export_trajectory_derivative_charts(traj_csv: str | Path = "results/data/trajectory_samples.csv", output_dir: str | Path = "images/generated") -> None:
    path = resolve_path(traj_csv)
    output = resolve_path(output_dir)
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or not _require_columns(df, ["scene_id", "planner", "trial_id", "sample_index", "q"]):
        return
    rows = []
    for (scene_id, planner, trial_id), grp in df.groupby(["scene_id", "planner", "trial_id"]):
        ordered = grp.sort_values("sample_index")
        q_values = []
        for value in ordered["q"]:
            try:
                q_values.append(_parse_q(value))
            except ValueError:
                continue
        if len(q_values) < 2:
            continue
        arr = np.asarray(q_values, dtype=float)
        speed = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        acc = np.linalg.norm(np.diff(arr, n=2, axis=0), axis=1) if len(arr) >= 3 else np.asarray([0.0])
        jerk = np.linalg.norm(np.diff(arr, n=3, axis=0), axis=1) if len(arr) >= 4 else np.asarray([0.0])
        rows.append({
            "scene_id": scene_id,
            "planner": planner,
            "trial_id": trial_id,
            "max_speed": float(np.max(speed)),
            "mean_speed": float(np.mean(speed)),
            "max_acc": float(np.max(acc)),
            "mean_acc": float(np.mean(acc)),
            "max_jerk": float(np.max(jerk)),
            "mean_jerk": float(np.mean(jerk)),
        })
    derived = pd.DataFrame(rows)
    if derived.empty:
        return
    derived_path = resolve_path("results/data/trajectory_derivative_summary.csv")
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(derived_path, index=False)
    for metric, ylabel, name in [
        ("max_speed", "Max joint speed increment", "trajectory_derived_speed_boxplot"),
        ("max_acc", "Max joint acceleration increment", "trajectory_derived_acc_boxplot"),
        ("max_jerk", "Max joint jerk increment", "trajectory_derived_jerk_boxplot"),
    ]:
        _boxplot(derived, "planner", metric, f"Derived trajectory {metric} distribution", ylabel, output, name)


def export_all_plots(config_path: str | Path | None = None) -> None:
    from arm_planning.experiments.runner import write_summary_tables

    write_summary_tables()
    export_mujoco_scene_screenshots(config_path=config_path)
    export_mujoco_failure_snapshots(config_path=config_path)
    export_ik_charts()
    export_planner_charts()
    export_trajectory_charts()
    export_trajectory_metric_charts()
    export_paper_trajectory_figures(config_path=config_path)
    export_extended_ik_charts()
    export_extended_planner_charts()
    export_trajectory_derivative_charts()
