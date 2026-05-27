from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from arm_planning.utils.config import resolve_path


class ResultStore:
    def __init__(self, data_dir: str | Path = "results/data"):
        self.data_dir = resolve_path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.ik_path = self.data_dir / "ik_results.csv"
        self.planner_path = self.data_dir / "planner_results.csv"
        self.trajectory_path = self.data_dir / "trajectory_samples.csv"
        self.ik_summary_path = self.data_dir / "ik_summary.csv"
        self.planner_summary_path = self.data_dir / "planner_summary.csv"
        self.planner_failure_summary_path = self.data_dir / "planner_failure_summary.csv"
        self.planner_failure_snapshots_path = self.data_dir / "planner_failure_snapshots.csv"
        self.planner_ranking_path = self.data_dir / "planner_ranking.csv"
        self.trajectory_metrics_path = self.data_dir / "trajectory_metrics.csv"
        self.reproducibility_path = self.data_dir / "reproducibility_report.csv"

    def reset(self) -> None:
        for path in [
            self.ik_path,
            self.planner_path,
            self.trajectory_path,
            self.ik_summary_path,
            self.planner_summary_path,
            self.planner_failure_summary_path,
            self.planner_failure_snapshots_path,
            self.planner_ranking_path,
            self.trajectory_metrics_path,
            self.reproducibility_path,
        ]:
            if path.exists():
                path.unlink()

    def append_row(self, path: Path, row: dict) -> None:
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def save_ik_result(self, row: dict) -> None:
        self.append_row(self.ik_path, row)

    def save_planner_result(self, row: dict) -> None:
        self.append_row(self.planner_path, row)

    def save_planner_failure_snapshot(self, row: dict) -> None:
        self.append_row(self.planner_failure_snapshots_path, row)

    def save_trajectory(self, rows: Iterable[dict]) -> None:
        for row in rows:
            self.append_row(self.trajectory_path, row)

    def save_trajectory_metrics(self, row: dict) -> None:
        self.append_row(self.trajectory_metrics_path, row)

    def save_reproducibility_row(self, row: dict) -> None:
        self.append_row(self.reproducibility_path, row)


def array_to_string(value: np.ndarray | list[float]) -> str:
    arr = np.asarray(value, dtype=float)
    return "[" + ",".join(f"{x:.8g}" for x in arr.tolist()) + "]"
