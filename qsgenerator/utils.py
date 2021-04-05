import math
from dataclasses import dataclass

import cirq
import numpy as np
from typing import List


@dataclass
class FidelityGrid:
    prob_gen: float
    label_gen: any
    prob_real: float
    label_real: any
    fidelity: float
    abs_fidelity: float


@dataclass
class GeneratorsFidelityGrid:
    label_gen1: any
    label_gen2: any
    fidelity: float
    abs_fidelity: float


def map_to_radians(val, low=-1, high=1):
    span = high - low
    norm_val = float(val - low) / float(span)
    return norm_val * math.pi


def get_zero_ones_array(size: int, one_indices: List[int]) -> np.ndarray:
    res = np.zeros(size)
    res[one_indices] = 1
    return res


def get_fidelity_grid(generated, real):
    return [FidelityGrid(pg, lg, pr, lr, cirq.fidelity(g, r), cirq.fidelity(abs_g, abs_r))
            for pr, lr, r, abs_r in real for pg, lg, g, abs_g in generated]


def get_generator_fidelity_grid(generated):
    return [GeneratorsFidelityGrid(generated[i][1],
                                   generated[j][1],
                                   cirq.fidelity(generated[i][2], generated[j][2]),
                                   cirq.fidelity(generated[i][3], generated[j][3]))
            for i in range(len(generated)) for j in range(len(generated)) if i != j]
