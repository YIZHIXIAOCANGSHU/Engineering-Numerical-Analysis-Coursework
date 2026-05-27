from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_trial_seed(global_seed: int, scene_index: int, algorithm_index: int, trial_id: int) -> int:
    return int(global_seed + scene_index * 10000 + algorithm_index * 1000 + trial_id)


def create_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))
