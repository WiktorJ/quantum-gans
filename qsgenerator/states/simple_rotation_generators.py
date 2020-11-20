from typing import Dict

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
