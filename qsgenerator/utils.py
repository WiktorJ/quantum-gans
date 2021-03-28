import math
import numpy as np
from typing import List


def map_to_radians(val, low=-1, high=1):
    span = high - low
    norm_val = float(val - low) / float(span)
    return norm_val * math.pi


def get_zero_ones_array(size: int, one_indices: List[int]) -> np.ndarray:
    res = np.zeros(size)
    res[one_indices] = 1
    return res
