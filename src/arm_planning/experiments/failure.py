from __future__ import annotations


FAILURE_CATEGORY_CN = {
    "success": "成功",
    "ik_failed": "IK 未收敛",
    "invalid_start_goal": "起点/目标无效",
    "ground_penetration": "地面穿透",
    "obstacle_collision": "障碍碰撞",
    "roadmap_disconnected": "采样图未连通",
    "local_minimum": "局部极小值",
    "iteration_limit": "迭代上限",
    "postprocess_invalid": "后处理轨迹无效",
    "other": "其他",
}


def failure_category(success: bool, message: str = "", phase: str = "") -> tuple[str, str]:
    if bool(success):
        key = "success"
    else:
        text = f"{phase} {message}".lower()
        phase_text = phase.lower()
        if "ground" in text or "floor" in text or "地面" in text:
            key = "ground_penetration"
        elif phase_text == "ik" or ("ik" in text and ("failed" in text or "max iterations" in text or "not converged" in text)):
            key = "ik_failed"
        elif "postprocess" in text or "post-processing" in text:
            key = "postprocess_invalid"
        elif "roadmap" in text or "no roadmap path" in text:
            key = "roadmap_disconnected"
        elif "local minimum" in text:
            key = "local_minimum"
        elif "max iterations" in text or "iteration" in text:
            key = "iteration_limit"
        elif phase_text == "validation" or "start" in text or "goal" in text:
            key = "invalid_start_goal"
        elif "collision" in text or "obstacle" in text or "blocked" in text:
            key = "obstacle_collision"
        elif "invalid" in text:
            key = "invalid_start_goal"
        else:
            key = "other"
    return key, FAILURE_CATEGORY_CN[key]
