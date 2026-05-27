from __future__ import annotations

import time

import mujoco
import numpy as np
from scipy.optimize import least_squares

from arm_planning.core.math_utils import clip_to_limits
from arm_planning.core.types import IKResult
from arm_planning.ik.base import IKSolver
from arm_planning.sim.mujoco_world import MujocoWorld


def _normalize_quat(quat: np.ndarray | None) -> np.ndarray | None:
    if quat is None:
        return None
    q = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("target_quat_wxyz must be non-zero")
    q = q / norm
    if q[0] < 0.0:
        q *= -1.0
    return q


def orientation_error_vector(current_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    current = _normalize_quat(current_quat)
    target = _normalize_quat(target_quat)
    assert current is not None and target is not None
    inv_current = np.asarray([current[0], -current[1], -current[2], -current[3]], dtype=float)
    diff = np.zeros(4, dtype=float)
    mujoco.mju_mulQuat(diff, target, inv_current)
    if diff[0] < 0.0:
        diff *= -1.0
    angle = 2.0 * np.arctan2(float(np.linalg.norm(diff[1:])), float(np.clip(diff[0], -1.0, 1.0)))
    if angle <= 1e-12:
        return np.zeros(3, dtype=float)
    return diff[1:] / max(float(np.linalg.norm(diff[1:])), 1e-12) * angle


class IterativeJacobianIK(IKSolver):
    def __init__(
        self,
        name: str,
        max_iterations: int = 160,
        tolerance: float = 0.012,
        step_scale: float = 0.65,
        damping: float = 0.08,
        orientation_tolerance: float = 0.12,
        orientation_weight: float = 0.35,
    ):
        self.name = name
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.step_scale = float(step_scale)
        self.damping = float(damping)
        self.orientation_tolerance = float(orientation_tolerance)
        self.orientation_weight = float(orientation_weight)

    def _delta(self, jac: np.ndarray, error: np.ndarray, iteration: int) -> np.ndarray:
        if self.name == "pinv":
            return np.linalg.pinv(jac) @ error
        if self.name == "dls":
            lam = self.damping
        elif self.name == "lm":
            lam = self.damping * (1.0 + 0.01 * iteration)
        else:
            raise ValueError(f"Unknown iterative IK method: {self.name}")
        lhs = jac @ jac.T + (lam * lam) * np.eye(jac.shape[0])
        return jac.T @ np.linalg.solve(lhs, error)

    def solve(
        self,
        target_position: np.ndarray,
        q_seed: np.ndarray,
        context: MujocoWorld,
        target_quat_wxyz: np.ndarray | None = None,
    ) -> IKResult:
        start = time.perf_counter()
        target = np.asarray(target_position, dtype=float)
        target_quat = _normalize_quat(target_quat_wxyz)
        q = clip_to_limits(np.asarray(q_seed, dtype=float).copy(), context.lower_limits, context.upper_limits)
        message = "max iterations reached"
        success = False
        iterations = 0
        for iteration in range(self.max_iterations):
            current = context.forward_kinematics(q)
            pos_error = target - current
            orient_error = np.zeros(3, dtype=float) if target_quat is None else orientation_error_vector(context.get_ee_quat(), target_quat)
            error = pos_error if target_quat is None else np.concatenate([pos_error, self.orientation_weight * orient_error])
            err_norm = float(np.linalg.norm(error))
            iterations = iteration + 1
            if float(np.linalg.norm(pos_error)) <= self.tolerance and (target_quat is None or float(np.linalg.norm(orient_error)) <= self.orientation_tolerance):
                success = True
                message = "converged"
                break
            if target_quat is None:
                jac = context.compute_jacobian(q)
            else:
                jacp, jacr = context.compute_pose_jacobian(q)
                jac = np.vstack([jacp, self.orientation_weight * jacr])
            try:
                delta = self._delta(jac, error, iteration)
            except np.linalg.LinAlgError:
                message = "linear solve failed"
                break
            max_step = 0.35
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > max_step:
                delta = delta / delta_norm * max_step
            q = clip_to_limits(q + self.step_scale * delta, context.lower_limits, context.upper_limits)
        final_error = float(np.linalg.norm(target - context.forward_kinematics(q)))
        final_orientation_error = 0.0 if target_quat is None else float(np.linalg.norm(orientation_error_vector(context.get_ee_quat(), target_quat)))
        return IKResult(
            success=bool(success or (final_error <= self.tolerance and (target_quat is None or final_orientation_error <= self.orientation_tolerance))),
            q=q.copy(),
            position_error=final_error,
            orientation_error=final_orientation_error,
            iterations=iterations,
            solve_time=float(time.perf_counter() - start),
            method=self.name,
            condition_number=context.condition_number(q),
            message=message,
        )


class ScipyLeastSquaresIK(IKSolver):
    name = "scipy_baseline"

    def __init__(self, max_iterations: int = 160, tolerance: float = 0.012, orientation_tolerance: float = 0.12, orientation_weight: float = 0.35):
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.orientation_weight = float(orientation_weight)

    def solve(
        self,
        target_position: np.ndarray,
        q_seed: np.ndarray,
        context: MujocoWorld,
        target_quat_wxyz: np.ndarray | None = None,
    ) -> IKResult:
        start = time.perf_counter()
        target = np.asarray(target_position, dtype=float)
        target_quat = _normalize_quat(target_quat_wxyz)

        def residual(q: np.ndarray) -> np.ndarray:
            position_residual = context.forward_kinematics(q) - target
            if target_quat is None:
                return position_residual
            orientation_residual = -orientation_error_vector(context.get_ee_quat(), target_quat)
            return np.concatenate([position_residual, self.orientation_weight * orientation_residual])

        result = least_squares(
            residual,
            x0=clip_to_limits(q_seed, context.lower_limits, context.upper_limits),
            bounds=(context.lower_limits, context.upper_limits),
            method="trf",
            max_nfev=self.max_iterations,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )
        q = result.x.copy()
        context.forward_kinematics(q)
        final_position_error = float(np.linalg.norm(context.get_ee_position() - target))
        final_orientation_error = 0.0 if target_quat is None else float(np.linalg.norm(orientation_error_vector(context.get_ee_quat(), target_quat)))
        return IKResult(
            success=bool(result.success and final_position_error <= self.tolerance and (target_quat is None or final_orientation_error <= self.orientation_tolerance)),
            q=q,
            position_error=final_position_error,
            orientation_error=final_orientation_error,
            iterations=int(result.nfev),
            solve_time=float(time.perf_counter() - start),
            method=self.name,
            condition_number=context.condition_number(q),
            message=str(result.message),
        )


def create_ik_solver(name: str, params: dict) -> IKSolver:
    if name in {"pinv", "dls", "lm"}:
        return IterativeJacobianIK(
            name=name,
            max_iterations=params.get("max_iterations", 160),
            tolerance=params.get("tolerance", 0.012),
            step_scale=params.get("step_scale", 0.65),
            damping=params.get("damping", 0.08),
            orientation_tolerance=params.get("orientation_tolerance_rad", 0.12),
            orientation_weight=params.get("orientation_weight", 0.35),
        )
    if name == "scipy_baseline":
        return ScipyLeastSquaresIK(
            max_iterations=params.get("max_iterations", 160),
            tolerance=params.get("tolerance", 0.012),
            orientation_tolerance=params.get("orientation_tolerance_rad", 0.12),
            orientation_weight=params.get("orientation_weight", 0.35),
        )
    raise ValueError(f"Unknown IK solver: {name}")
