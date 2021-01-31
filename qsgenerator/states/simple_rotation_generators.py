from typing import Dict, List

import numpy as np


def get_binary_x_rotation_provider(binary_mappings: Dict[float, str]):
    def provider(g: float):
        return rotations[g]

    rotations = {}
    for g, b_string in binary_mappings.items():
        rotation = []
        for b in b_string:
            if b == '0':
                rotation.append(0)
            else:
                rotation.append(np.pi)
        rotations[g] = rotation

    return provider


def get_arcsin_x_rotation_provider(bases: List[float], circuit_width: int):
    def provider(g: float):
        return rotations[g]

    step = 0.1
    rotations = {}
    for b in bases:
        assert b >= -1 and b + (circuit_width * step) <= 1
        rotation = []
        x = b
        for _ in range(circuit_width):
            rotation.append(2 * np.arcsin(x))
            x += step
        rotations[b] = rotation

    return provider
